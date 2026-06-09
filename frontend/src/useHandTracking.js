import { useEffect, useRef } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

// ---- These MUST match collect_dataset.py so live input matches training ----
const CROP_SIZE = 224;
const MARGIN = 0.25;
const MIRROR = true; // match collect_dataset.py MIRROR flag
// ---------------------------------------------------------------------------
const HOLD_MS = 1000; // hold still ~1s to capture a letter
const MOVE_THRESHOLD = 0.04; // normalized movement counted as "still moving"

export function useHandTracking({ videoRef, enabled, onCapture, onStatus }) {
  // keep latest callbacks without restarting the camera every render
  const onCaptureRef = useRef(onCapture);
  const onStatusRef = useRef(onStatus);
  useEffect(() => {
    onCaptureRef.current = onCapture;
    onStatusRef.current = onStatus;
  });

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;
    let rafId = 0;
    let landmarker = null;
    let stream = null;

    let stableSince = 0;
    let lastCenter = null;
    let armed = true; // can capture? re-armed after the hand moves to a new pose

    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = CROP_SIZE;
    cropCanvas.height = CROP_SIZE;

    function makeCrop(minX, minY, maxX, maxY) {
      const video = videoRef.current;
      if (!video) return null;
      const vw = video.videoWidth;
      const vh = video.videoHeight;

      const pxMinX = minX * vw, pxMaxX = maxX * vw;
      const pxMinY = minY * vh, pxMaxY = maxY * vh;
      const bcx = (pxMinX + pxMaxX) / 2;
      const bcy = (pxMinY + pxMaxY) / 2;
      const side = Math.max(pxMaxX - pxMinX, pxMaxY - pxMinY) * (1 + MARGIN);
      const half = side / 2;

      const sx = Math.max(0, bcx - half);
      const sy = Math.max(0, bcy - half);
      const sw = Math.min(vw - sx, side);
      const sh = Math.min(vh - sy, side);

      const ctx = cropCanvas.getContext("2d");
      ctx.save();
      ctx.clearRect(0, 0, CROP_SIZE, CROP_SIZE);
      if (MIRROR) {
        ctx.translate(CROP_SIZE, 0);
        ctx.scale(-1, 1);
      }
      ctx.drawImage(video, sx, sy, sw, sh, 0, 0, CROP_SIZE, CROP_SIZE);
      ctx.restore();
      return cropCanvas.toDataURL("image/jpeg", 0.9);
    }

    function handleResult(result) {
      const hands = result?.landmarks;
      if (!hands || hands.length === 0) {
        onStatusRef.current?.("No hand");
        stableSince = 0;
        lastCenter = null;
        armed = true;
        return;
      }

      const lm = hands[0];
      let minX = 1, minY = 1, maxX = 0, maxY = 0;
      for (const p of lm) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      }
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      const now = performance.now();

      let moving = false;
      if (lastCenter) {
        const dist = Math.hypot(cx - lastCenter.x, cy - lastCenter.y);
        moving = dist > MOVE_THRESHOLD;
      }
      lastCenter = { x: cx, y: cy };

      if (moving) {
        stableSince = now;
        armed = true; // moved to a new pose -> allowed to capture again
        onStatusRef.current?.("Hold still to capture");
        return;
      }

      if (stableSince === 0) stableSince = now;
      const heldFor = now - stableSince;

      if (armed && heldFor >= HOLD_MS) {
        armed = false; // capture only once per hold
        const dataUrl = makeCrop(minX, minY, maxX, maxY);
        if (dataUrl) {
          onStatusRef.current?.("Captured");
          onCaptureRef.current?.(dataUrl);
        }
      } else if (armed) {
        onStatusRef.current?.("Hold…");
      }
    }

    function loop() {
      const video = videoRef.current;
      if (!cancelled && video && landmarker && video.readyState >= 2) {
        handleResult(landmarker.detectForVideo(video, performance.now()));
      }
      rafId = requestAnimationFrame(loop);
    }

    async function setup() {
      onStatusRef.current?.("Loading hand model…");
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );
      landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        numHands: 1,
        runningMode: "VIDEO",
      });
      if (cancelled) return;

      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (cancelled) return;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      onStatusRef.current?.("Show a sign");
      loop();
    }

    setup().catch((e) => {
      console.error(e);
      onStatusRef.current?.("Camera/model error: " + e.message);
    });

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
      if (stream) stream.getTracks().forEach((t) => t.stop());
      const video = videoRef.current;
      if (video) video.srcObject = null;
      if (landmarker) landmarker.close();
    };
  }, [enabled, videoRef]);
}
