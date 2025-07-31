import grpc

from proto_gen import blinky_pb2_grpc, blinky_pb2

class BlinkyClient:
    def __init__(self, server_address='192.168.3.232:50051'):
        self.server_address = server_address
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = blinky_pb2_grpc.BlinkServiceStub(self.channel)

    def SetLedOn(self, line:int, on:bool):
        try:
            request = blinky_pb2.SetLEDRequest(line=line, on=on)
            self.stub.SetLED(request)
            
        except Exception as e:
            print(f"Error: {e}")

    def IsLedOn(self, line:int) -> bool:
        try:
            request = blinky_pb2.IsOnRequest(line=line)
            response  = self.stub.IsOn(request)
            print(f"is led on: {response.is_on}")
            return response.is_on
        except Exception as e:
            print(f"Error: {e}")

    def close(self):
        self.channel.close()

# For testing purposes, you can run this script directly to get a sample IMU data.
if __name__ == "__main__":
    client = BlinkyClient()
    client.SetLedOn(1, True)
    client.close()