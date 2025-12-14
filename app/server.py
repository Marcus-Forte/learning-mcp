# server.py
from mcp.server.fastmcp import FastMCP

import asyncio

from proto_gen.control_pb2_grpc import RobotControlStub
from proto_gen.control_pb2 import MoveRequest, MoveDirection
from proto_gen.sensor_pb2_grpc import SensorServiceStub
from proto_gen.sensor_pb2 import AdcDataRequest
from enum import Enum
import grpc

class RobotDirection(Enum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3


async def main():
    # Stateless server (no session persistence, no sse stream with supported client)
    mcp = FastMCP("MCPServer", stateless_http=True, json_response=True)
    channel = grpc.insecure_channel("192.168.3.251:50051")
    robot_stub = RobotControlStub(channel)

    sensor_channel = grpc.insecure_channel("192.168.3.251:50054")
    sensor_stub = SensorServiceStub(sensor_channel) 

     # Test Move
    # robot_stub.Move(MoveRequest(direction=MoveDirection.MOVE_FORWARD, duration=1.0))

    @mcp.tool()
    def move_robot(direction: str, duration: float) -> str:
        match direction.strip().lower():
            case "forward":
                direction = MoveDirection.MOVE_FORWARD
            case "backward" | "back":
                direction = MoveDirection.MOVE_BACKWARD
            case "left":
                direction = MoveDirection.MOVE_LEFT
            case "right":
                direction = MoveDirection.MOVE_RIGHT
            case _:
                raise ValueError("Invalid direction")

        print(f"Moving robot {direction} for {duration} seconds")
        request = MoveRequest(direction= direction, duration=duration)
        robot_stub.Move(request)
        return "Success"

    @mcp.tool()
    def get_battery_level() -> str:
        request = AdcDataRequest(channel=0)
        response = sensor_stub.GetAdc(request)
        return f"Battery level is {response.sample} Volts"

    print("Starting MCP server...")
    await mcp.run_streamable_http_async()


if __name__ == "__main__":
    asyncio.run(main())
