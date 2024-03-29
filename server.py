import socket

from crypto import KeyManager, DES


class Server:
    def __init__(self, addr, port, buffer_size=1024):
        self.addr = addr
        self.port = port
        self.buffer_size = buffer_size

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.addr, self.port))
        self.s.listen(1)
        self.conn, self.addr = self.s.accept()
        print("server running")

    def send(self, msg_bytes: bytes):
        self.conn.send(msg_bytes)

    def recv(self, buffer_size=None) -> bytes:
        if buffer_size is None:
            buffer_size = self.buffer_size
        msg_bytes = self.conn.recv(buffer_size)

        return msg_bytes

    def close(self):
        self.conn.close()


if __name__ == '__main__':
    server = Server('localhost', 9999)
    key = KeyManager.read_key('key.txt')
    des = DES(key)

    while True:
        print("Server is running...")
        print("key is:", key.decode('utf-8'))

        encrypted_msg = server.recv()
        print("Received ciphertext:", encrypted_msg.hex())

        decrypted_msg = des.decrypt(encrypted_msg)
        print("Received plaintext:", decrypted_msg)

        msg = input('Type message: ')
        if msg == 'exit':
            break

        encrypted_response = des.encrypt(msg)
        print("send plaintext:", msg)
        print("send ciphertext:", encrypted_response.hex())
        server.send(encrypted_response)

        print("----------------------")

    server.close()
