import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import readline from "readline";
import { spawn } from "child_process";
import { fileURLToPath } from "url";
import sharp from "sharp";
import { AccessToken } from "livekit-server-sdk";
import { Room, RoomEvent, TrackKind, VideoStream, VideoBufferType } from "@livekit/rtc-node";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const LIVEKIT_URL = process.env.LIVEKIT_URL;
const LIVEKIT_API_KEY = process.env.LIVEKIT_API_KEY;
const LIVEKIT_API_SECRET = process.env.LIVEKIT_API_SECRET;

const ROOM_NAME = process.env.ROOM_NAME || "riya-interview-room";
const AGENT_IDENTITY = process.env.AGENT_IDENTITY || "riya-vision-agent";
const SAMPLE_INTERVAL_MS = Number(process.env.SAMPLE_INTERVAL_MS || 1000);
const EVENTS_LOG_PATH = new URL("./events.jsonl", import.meta.url);

const PYTHON_BIN = process.env.PYTHON_BIN || "python3";
const LOCAL_VISION_WORKER = process.env.LOCAL_VISION_WORKER || path.join(__dirname, "local_vision_worker.py");
const LOCAL_VISION_TIMEOUT_MS = Number(process.env.LOCAL_VISION_TIMEOUT_MS || 20000);

const VIOLATION_STATUSES = new Set(["no_person", "multiple_persons", "partial_frame", "low_confidence"]);
const SNAPSHOT_MIN_INTERVAL_MS = Number(process.env.SNAPSHOT_MIN_INTERVAL_MS || 4000);
const SNAPSHOT_WIDTH = Number(process.env.SNAPSHOT_WIDTH || 160);

function requireEnv(name, value) {
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
}

function logEvent(room, event) {
  const { snapshot_base64, ...loggable } = event;
  const line = JSON.stringify({ ...loggable, has_snapshot: !!snapshot_base64 }) + "\n";
  fs.appendFileSync(EVENTS_LOG_PATH, line);
  console.log(
    `[${event.timestamp}] source=${event.source || "local"} status=${event.status} count=${event.person_count} confidence=${event.confidence}${
      event.reason ? ` reason="${event.reason}"` : ""
    }`
  );

  room.localParticipant
    ?.publishData(new TextEncoder().encode(JSON.stringify(event)), {
      reliable: true,
      topic: "riya-vision-events",
    })
    .catch((err) => console.error("Failed to broadcast event:", err.message));
}

async function mintAgentToken() {
  requireEnv("LIVEKIT_API_KEY", LIVEKIT_API_KEY);
  requireEnv("LIVEKIT_API_SECRET", LIVEKIT_API_SECRET);

  const token = new AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET, {
    identity: AGENT_IDENTITY,
    ttl: "6h",
  });

  token.addGrant({
    roomJoin: true,
    room: ROOM_NAME,
    canPublish: false,
    canSubscribe: true,
    canPublishData: true,
  });

  return token.toJwt();
}

async function frameToJpegBase64(frame) {
  const rgba = frame.convert(VideoBufferType.RGBA);
  const jpeg = await sharp(Buffer.from(rgba.data), {
    raw: { width: rgba.width, height: rgba.height, channels: 4 },
  })
    .jpeg({ quality: 72 })
    .toBuffer();

  return jpeg.toString("base64");
}

async function frameToThumbnailBase64(frame) {
  const rgba = frame.convert(VideoBufferType.RGBA);
  const jpeg = await sharp(Buffer.from(rgba.data), {
    raw: { width: rgba.width, height: rgba.height, channels: 4 },
  })
    .resize({ width: SNAPSHOT_WIDTH })
    .jpeg({ quality: 55 })
    .toBuffer();

  return jpeg.toString("base64");
}

function normalizeDetection(detection) {
  const allowedStatuses = new Set([
    "ok",
    "no_person",
    "multiple_persons",
    "partial_frame",
    "low_confidence",
    "vision_error",
  ]);

  const status = allowedStatuses.has(detection?.status) ? detection.status : "vision_error";
  const personCount = Number.isInteger(detection?.person_count) ? detection.person_count : null;
  const confidence = Number.isFinite(Number(detection?.confidence))
    ? Math.max(0, Math.min(1, Number(detection.confidence)))
    : 0;
  const bbox = Array.isArray(detection?.bbox) && detection.bbox.length === 4
    ? detection.bbox.map((value) => Math.max(0, Math.min(1, Number(value) || 0)))
    : [];

  return {
    source: "local-dfine",
    status,
    person_count: personCount,
    confidence,
    bbox,
    reason: typeof detection?.reason === "string" ? detection.reason : "Local vision returned an invalid response.",
    boxes: Array.isArray(detection?.boxes) ? detection.boxes : [],
    inference_ms: Number.isFinite(Number(detection?.inference_ms)) ? Number(detection.inference_ms) : null,
  };
}

class LocalVisionClient {
  constructor() {
    this.nextId = 1;
    this.pending = new Map();
    this.process = null;
    this.readline = null;
    this.start();
  }

  start() {
    const args = [LOCAL_VISION_WORKER];

    // Keep the D-FINE settings in env so this agent remains easy to configure.
    const childEnv = {
      ...process.env,
      PYTHONUNBUFFERED: "1",
    };

    this.process = spawn(PYTHON_BIN, args, {
      cwd: __dirname,
      env: childEnv,
      stdio: ["pipe", "pipe", "pipe"],
    });

    this.process.stderr.on("data", (chunk) => {
      process.stderr.write(`[local-vision] ${chunk}`);
    });

    this.readline = readline.createInterface({ input: this.process.stdout });
    this.readline.on("line", (line) => this.handleLine(line));

    this.process.on("exit", (code, signal) => {
      const message = `Local vision worker exited with code=${code} signal=${signal}`;
      console.error(message);
      for (const { reject, timeout } of this.pending.values()) {
        clearTimeout(timeout);
        reject(new Error(message));
      }
      this.pending.clear();
      this.process = null;
    });

    this.process.on("error", (error) => {
      console.error("Failed to start local vision worker:", error.message);
    });
  }

  handleLine(line) {
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      console.error("Invalid JSON from local vision worker:", line);
      return;
    }

    const pending = this.pending.get(message.id);
    if (!pending) return;

    clearTimeout(pending.timeout);
    this.pending.delete(message.id);

    if (message.error) {
      pending.reject(new Error(message.error));
      return;
    }

    pending.resolve(normalizeDetection(message.result));
  }

  async detect(base64Jpeg) {
    if (!this.process || this.process.killed || !this.process.stdin.writable) {
      throw new Error("Local vision worker is not running.");
    }

    const id = this.nextId++;
    const payload = JSON.stringify({ id, image_base64: base64Jpeg }) + "\n";

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Local vision timed out after ${LOCAL_VISION_TIMEOUT_MS}ms`));
      }, LOCAL_VISION_TIMEOUT_MS);

      this.pending.set(id, { resolve, reject, timeout });
      this.process.stdin.write(payload, (error) => {
        if (error) {
          clearTimeout(timeout);
          this.pending.delete(id);
          reject(error);
        }
      });
    });
  }

  stop() {
    for (const { reject, timeout } of this.pending.values()) {
      clearTimeout(timeout);
      reject(new Error("Local vision worker stopped."));
    }
    this.pending.clear();

    if (this.readline) this.readline.close();
    if (this.process && !this.process.killed) {
      this.process.kill("SIGTERM");
    }
  }
}

async function handleVideoTrack(room, vision, track, participantIdentity) {
  console.log(`Subscribed to video track from participant: ${participantIdentity}`);

  const stream = new VideoStream(track);
  let nextAllowedAt = 0;
  let processing = false;
  let lastViolationStatus = null;
  let lastSnapshotAt = 0;

  for await (const { frame } of stream) {
    const now = Date.now();
    if (now < nextAllowedAt || processing) continue;

    nextAllowedAt = now + SAMPLE_INTERVAL_MS;
    processing = true;

    try {
      const base64Jpeg = await frameToJpegBase64(frame);
      const detection = await vision.detect(base64Jpeg);

      let snapshotBase64 = null;
      if (VIOLATION_STATUSES.has(detection.status)) {
        const isNewViolation = detection.status !== lastViolationStatus;
        const dueForHeartbeat = now - lastSnapshotAt >= SNAPSHOT_MIN_INTERVAL_MS;
        if (isNewViolation || dueForHeartbeat) {
          snapshotBase64 = await frameToThumbnailBase64(frame);
          lastSnapshotAt = now;
        }
        lastViolationStatus = detection.status;
      } else {
        lastViolationStatus = null;
      }

      logEvent(room, {
        timestamp: new Date(now).toISOString(),
        participant: participantIdentity,
        ...detection,
        snapshot_base64: snapshotBase64,
      });
    } catch (error) {
      console.error("Local detection error:", error.message);
      logEvent(room, {
        timestamp: new Date(now).toISOString(),
        participant: participantIdentity,
        source: "local-dfine",
        status: "vision_error",
        person_count: null,
        confidence: 0,
        bbox: [],
        boxes: [],
        inference_ms: null,
        reason: error.message,
        error: error.message,
      });
    } finally {
      processing = false;
    }
  }
}

async function main() {
  requireEnv("LIVEKIT_URL", LIVEKIT_URL);
  requireEnv("LIVEKIT_API_KEY", LIVEKIT_API_KEY);
  requireEnv("LIVEKIT_API_SECRET", LIVEKIT_API_SECRET);

  const vision = new LocalVisionClient();
  const token = await mintAgentToken();
  const room = new Room();

  const shutdown = () => {
    console.log("Shutting down local vision agent...");
    vision.stop();
    room.disconnect();
    process.exit(0);
  };

  process.once("SIGINT", shutdown);
  process.once("SIGTERM", shutdown);

  room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
    if (track.kind === TrackKind.KIND_VIDEO) {
      handleVideoTrack(room, vision, track, participant.identity).catch((err) =>
        console.error("Video handler crashed:", err)
      );
    }
  });

  room.on(RoomEvent.Disconnected, () => {
    console.log("Agent disconnected from room");
  });

  await room.connect(LIVEKIT_URL, token, { autoSubscribe: true, dynacast: false });
  console.log(
    `Riya local vision agent connected to room "${ROOM_NAME}", sampling every ${SAMPLE_INTERVAL_MS}ms`
  );
}

main().catch((err) => {
  console.error("Agent failed to start:", err);
  process.exit(1);
});
