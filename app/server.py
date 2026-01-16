# server.py
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP

import asyncio

from proto_gen.control_pb2_grpc import RobotControlStub
from proto_gen.control_pb2 import MoveRequest, MoveDirection
from proto_gen.sensor_pb2_grpc import SensorServiceStub
from proto_gen.sensor_pb2 import AdcDataRequest, CameraStreamRequest, CameraEncoding
from enum import Enum
import grpc
import threading
import time
import base64
import io
from PIL import Image


@dataclass
class ImageFrame:
    width: int
    height: int
    encoding: str
    timestamp: float | None
    received_at: float
    image_base64: bytes


class RobotDirection(Enum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3


async def main():
    # Stateless server (no session persistence, no sse stream with supported client)
    mcp = FastMCP("MCPServer", stateless_http=True, json_response=True)
    channel = grpc.insecure_channel("192.168.3.251:50052")
    robot_stub = RobotControlStub(channel)

    sensor_channel = grpc.insecure_channel("192.168.3.251:50051")
    sensor_stub = SensorServiceStub(sensor_channel) 

    latest_camera_frame = {
        "reply": None,
        "updated_at": None,
    }
    latest_camera_lock = threading.Lock()

    def camera_stream_worker() -> None:
        while True:
            try:
                stream = sensor_stub.getCameraFrame(CameraStreamRequest())
                print("Started camera stream")
                for reply in stream:
                    with latest_camera_lock:
                        latest_camera_frame["reply"] = reply
                        latest_camera_frame["updated_at"] = time.time()
            except Exception as exc:  # noqa: BLE001
                print(f"Camera stream error: {exc}")
                time.sleep(1.0)

    threading.Thread(target=camera_stream_worker).start()

    # Test Move
    # robot_stub.Move(MoveRequest(direction=MoveDirection.MOVE_FORWARD, duration=1.0))

    @mcp.tool()
    def move_robot(direction: str, duration: float) -> str:
        match direction.strip().lower():
            case "forward":
                direction = MoveDirection.MOVE_FORWARD
            case "backward" | "back" | "backwards":
                direction = MoveDirection.MOVE_BACKWARD
            case "left":
                direction = MoveDirection.MOVE_LEFT
            case "right":
                direction = MoveDirection.MOVE_RIGHT
            case "stop":
                pass
            case _:
                raise ValueError("Invalid direction")

        print(f"Moving robot {direction} for {duration} seconds")
        request = MoveRequest(direction= direction, duration=duration)
        robot_stub.Move(request)
        return "Success"

    @mcp.tool()
    def get_battery_level() -> str:
        request = AdcDataRequest(channel=0)
        response = sensor_stub.getAdcData(request)
        return f"Battery level is {response.sample} Volts"

    @mcp.tool()
    def get_camera_frame() -> ImageFrame:
        with latest_camera_lock:
            reply = latest_camera_frame["reply"]
            updated_at = latest_camera_frame["updated_at"]

        if reply is None:
            return "No camera frames received yet"

        encoding_name = CameraEncoding.Name(reply.encoding)

        image = Image.open(io.BytesIO(reply.image_data))
        image.save("debug_image.jpg", format="JPEG")

        image_b64 = base64.b64encode(reply.image_data).decode("utf-8")

        print("get camera")

        with open("debug_image.txt", "w", encoding="utf-8") as f:
            f.write(image_b64)

        return ImageFrame(
            width=reply.width,
            height=reply.height,
            encoding=encoding_name,
            timestamp=reply.timestamp if reply.HasField("timestamp") else None,
            received_at=updated_at,
            image_base64=image_b64,
        )

    print("Starting MCP server...")
    await mcp.run_streamable_http_async()


if __name__ == "__main__":
    asyncio.run(main())
