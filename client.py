import socket
from crypto import KeyManager, DES


class Client:
    def __init__(self, addr, port, buffer_size=1024):
        self.addr = addr
        self.port = port
        self.buffer_size = buffer_size

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.addr, self.port))

    def send(self, msg_bytes: bytes):
        self.s.send(msg_bytes)

    def recv(self, buffer_size=None) -> bytes:
        if buffer_size is None:
            buffer_size = self.buffer_size
        msg_bytes = self.s.recv(self.buffer_size)

        return msg_bytes

    def close(self):
        self.s.close()


if __name__ == '__main__':
    client = Client('localhost', 9999)
    key = KeyManager().read_key('key.txt')
    des = DES(key)

    while True:
        msg = input('Type message: ')
        if msg == 'exit':
            break

        print("Server is running...")
        print("key is:", key.decode('utf-8'))

        encrypted_msg = des.encrypt(msg)
        client.send(encrypted_msg)
        print("Send plaintext:", msg)
        print("Send ciphertext:", encrypted_msg.hex())

        server_response = client.recv()
        print("Received ciphertext:", server_response.hex())

        decrypted_response = des.decrypt(server_response)
        print("Received plaintext:", decrypted_response)

        print("----------------------")

    client.close()
