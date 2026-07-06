import dotenv from "dotenv";
import fs from "fs";
import sharp from "sharp";
import { AccessToken } from "livekit-server-sdk";
import { Room, RoomEvent, TrackKind, VideoStream, VideoBufferType } from "@livekit/rtc-node";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const LIVEKIT_URL = process.env.LIVEKIT_URL;
const LIVEKIT_API_KEY = process.env.LIVEKIT_API_KEY;
const LIVEKIT_API_SECRET = process.env.LIVEKIT_API_SECRET;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const ROOM_NAME = process.env.ROOM_NAME || "riya-interview-room";
const AGENT_IDENTITY = "riya-vision-agent";
const SAMPLE_INTERVAL_MS = Number(process.env.SAMPLE_INTERVAL_MS || 10000); // conservative default to fit free-tier daily quota
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash-lite";
const EVENTS_LOG_PATH = new URL("./events.jsonl", import.meta.url);

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

const detectionSchema = {
  type: "object",
  properties: {
    person_count: { type: "integer" },
    status: {
      type: "string",
      enum: ["ok", "no_person", "multiple_persons", "partial_frame"],
    },
    confidence: { type: "number" },
    bbox: {
      type: "array",
      items: { type: "number" },
      description: "[x_min, y_min, x_max, y_max] normalized 0-1 for the primary person, or empty array if none",
    },
    reason: {
      type: "string",
      description: "One short sentence explaining the status, e.g. what's out of frame or why multiple people were flagged",
    },
  },
  required: ["person_count", "status", "confidence", "bbox", "reason"],
};

const DETECTION_PROMPT = `You are a proctoring vision system for a live video interview.
Look at this single video frame and report:
- how many people are visible
- whether the primary candidate is fully in frame, partially out of frame, or absent
- your confidence (0-1)
- a bounding box for the primary person, normalized to 0-1 (x_min, y_min, x_max, y_max), or [] if no person
- a one-sentence reason explaining the status (e.g. "top of head cropped by frame", "second person visible in background", "candidate not visible")

status must be one of: "ok" (one person, fully in frame), "no_person", "multiple_persons", "partial_frame".
Respond with JSON only, matching the required schema.`;

function logEvent(room, event) {
  const line = JSON.stringify(event) + "\n";
  fs.appendFileSync(EVENTS_LOG_PATH, line);
  console.log(
    `[${event.timestamp}] status=${event.status} count=${event.person_count} confidence=${event.confidence}${
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
    .jpeg({ quality: 70 })
    .toBuffer();
  return jpeg.toString("base64");
}

function isOverloadedError(error) {
  try {
    return JSON.parse(error.message)?.error?.code === 503;
  } catch {
    return false;
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function detectPerson(base64Jpeg, retriesLeft = 1) {
  try {
    const response = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: [
        {
          role: "user",
          parts: [
            { text: DETECTION_PROMPT },
            { inlineData: { mimeType: "image/jpeg", data: base64Jpeg } },
          ],
        },
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: detectionSchema,
      },
    });

    return JSON.parse(response.text);
  } catch (error) {
    if (retriesLeft > 0 && isOverloadedError(error)) {
      await sleep(1200);
      return detectPerson(base64Jpeg, retriesLeft - 1);
    }
    throw error;
  }
}

function isRateLimitError(error) {
  try {
    return JSON.parse(error.message)?.error?.code === 429;
  } catch {
    return false;
  }
}

function parseRetryDelayMs(error, fallbackMs = 30000) {
  try {
    const details = JSON.parse(error.message)?.error?.details ?? [];
    const retryInfo = details.find((d) => d["@type"]?.includes("RetryInfo"));
    const match = /^(\d+(?:\.\d+)?)s$/.exec(retryInfo?.retryDelay ?? "");
    if (match) return Math.ceil(parseFloat(match[1]) * 1000);
  } catch {
    // fall through to default
  }
  return fallbackMs;
}

function describeError(error) {
  try {
    const parsed = JSON.parse(error.message)?.error;
    if (parsed?.code === 429) return "Rate limited: daily free-tier quota exceeded";
    if (parsed?.code === 503) return "Gemini temporarily overloaded, retrying";
    if (parsed?.message) return parsed.message;
  } catch {
    // not a structured API error, fall through
  }
  return error.message;
}

async function handleVideoTrack(room, track, participantIdentity) {
  console.log(`Subscribed to video track from participant: ${participantIdentity}`);

  const stream = new VideoStream(track);
  let nextAllowedAt = 0;
  let processing = false;

  for await (const { frame } of stream) {
    const now = Date.now();
    if (now < nextAllowedAt || processing) continue;
    nextAllowedAt = now + SAMPLE_INTERVAL_MS;
    processing = true;

    try {
      const base64Jpeg = await frameToJpegBase64(frame);
      const detection = await detectPerson(base64Jpeg);
      logEvent(room, {
        timestamp: new Date(now).toISOString(),
        participant: participantIdentity,
        ...detection,
      });
    } catch (error) {
      if (isRateLimitError(error)) {
        const backoffMs = parseRetryDelayMs(error);
        nextAllowedAt = now + backoffMs;
        console.error(`Rate limited, backing off for ${Math.round(backoffMs / 1000)}s`);
      } else {
        console.error("Detection error:", error.message);
      }
      logEvent(room, {
        timestamp: new Date(now).toISOString(),
        participant: participantIdentity,
        status: "error",
        person_count: null,
        confidence: 0,
        bbox: [],
        reason: describeError(error),
        error: error.message,
      });
    } finally {
      processing = false;
    }
  }
}

async function main() {
  const token = await mintAgentToken();
  const room = new Room();

  room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
    if (track.kind === TrackKind.KIND_VIDEO) {
      handleVideoTrack(room, track, participant.identity).catch((err) =>
        console.error("Video handler crashed:", err)
      );
    }
  });

  room.on(RoomEvent.Disconnected, () => {
    console.log("Agent disconnected from room");
  });

  await room.connect(LIVEKIT_URL, token, { autoSubscribe: true, dynacast: false });
  console.log(`Riya vision agent connected to room "${ROOM_NAME}", sampling every ${SAMPLE_INTERVAL_MS}ms`);
}

main().catch((err) => {
  console.error("Agent failed to start:", err);
  process.exit(1);
});
