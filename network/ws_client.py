import json
import asyncio
import websockets

from config import API_WS_HOST, API_WS_PORT

_RECONNECT_DELAY_S= 2.0
_URI= f"ws://{API_WS_HOST}:{API_WS_PORT}/ws/live/ingest"


async def websocket_session(bridge, send_queue: asyncio.Queue) -> None:
    while True:
        try:
            print(f"[ws_client] connecting to {_URI}")
            async with websockets.connect(_URI) as ws:
                print("[ws_client] connected")
                await asyncio.gather(
                    _receiver(ws, bridge),
                    _sender(ws, send_queue),
                )
        except Exception as e:
            print(f"[ws_client] connection failed: {e}")
            await asyncio.sleep(_RECONNECT_DELAY_S)


async def _receiver(ws, bridge) -> None:
    async for raw in ws:
        try:
            message= json.loads(raw)
            print(f"[ws_client] received: {message}")
            if message.get("type") == "prediction":
                print(f"[ws_client] emitting: {message['gesture']} {message['confidence']:.3f}")
                bridge.prediction_ready.emit(
                    message["gesture"],
                    float(message["confidence"]),
                )
        except Exception as e:
            print(f"[ws_client] error: {e}")


async def _sender(ws, send_queue: asyncio.Queue) -> None:
    while True:
        payload= await send_queue.get()
        if payload is None:
            return
        try:
            await ws.send(payload)
        except Exception:
            return