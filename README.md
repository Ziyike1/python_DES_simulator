# python_DES_simulator
implement the basic version of DES by python
## KeyManager Class
- **`read_key(key_file: str) -> bytes`**: Reads a key from a file and returns it as bytes.
- **`save_key(key_file: str, key: bytes)`**: Saves a key to a file.
- **`generate_key(key_len=256) -> bytes`**: Generates a random key of the specified length in bits and returns it as bytes.

### Utility Functions
- **`bitize(byts: bytes) -> list[int]`**: Converts bytes into a list of bits.
- **`debitize(bits: Iterable[int]) -> bytes`**: Converts a list of bits into bytes.
- **`bit2hex(bits: Iterable[int]) -> str`**: Converts bits into a hexadecimal string.
- **`hex2bit(hex_str: str) -> list[int]`**: Converts a hexadecimal string into bits.

### DES Class
- **`key_generation(key: list[int]) -> list[list[int]]`**: Generates 16 round subkeys from the initial 64-bit key.
#### Encryption Functions
- **`enc_block(block: list[int]) -> list[int]`**: Encrypts a 64-bit block using DES.
- **`mixer(L: list[int], R: list[int], sub_key: list[int]) -> list[int]`**: Mixes the left and right halves of the block.
- **`swapper(L: list[int], R: list[int]) -> tuple[list[int]]`**: Swaps the left and right halves of the block.
#### Decryption Functions
- **`dec_block(block: list[int]) -> list[int]`**: Decrypts a 64-bit block using DES.
#### Main Encryption and Decryption Methods
- **`encrypt(msg_str: str) -> bytes`**: Encrypts the entire message, padding if necessary.
- **`decrypt(msg_bytes: bytes) -> str`**: Decrypts the entire message, removing padding if necessary.

### Main Program Flow
1. **Initialization**: 
   - Create an instance of the `KeyManager` to read or generate keys.
   - Instantiate the `DES` class with the key.
2. **Key Generation**: Generate the round subkeys using the `key_generation` method.
3. **Encryption** (Server):
   - Receive a message from the client.
   - Encrypt the message using `encrypt` method.
   - Send the ciphertext back to the client.
4. **Decryption** (Client):
   - Receive the ciphertext from the server.
   - Decrypt the ciphertext using the `decrypt` method.
   - Display the decrypted message.

### Error Handling
- Properly handle exceptions such as key length, block size, and input validation.


