import secrets
from copy import deepcopy
import random
from typing import Iterable


class KeyManager:
    @staticmethod
    def read_key(key_file: str) -> bytes:
        with open(key_file, 'rb') as f:
            return f.read()

    @staticmethod
    def save_key(key_file: str, key: bytes):
        with open(key_file, 'wb') as f:
            f.write(key)

    def __init__(self, seed=None):
        self.random = random.Random(seed)

    def generate_key(self, key_len=256) -> bytes:
        """"
        Generate a random key of length key_len (bit length).
        return: random bytes of length (key_len // 8)
        """
        # TODO: your code here
        # Calculate the byte length required for the key
        byte_len = key_len // 8

        # Generate random bytes using secrets.token_bytes
        rand_bytes = secrets.token_bytes(byte_len)

        return rand_bytes


def bitize(byts: bytes) -> 'list[int]':
    """
    bitize bytes
    """
    bits = []
    # TODO: your code here
    # Iterate through each byte in the input bytes
    for b in byts:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)

    return bits


def debitize(bits: Iterable[int]) -> bytes:
    """
    debbitize a list of bits
    """
    if len(bits) % 8 != 0:
        raise ValueError('bits length is not a multiple of 8')

    byts = []

    # TODO: your code here
    # Iterate through the bits in chunks of 8 to form bytes
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        byts.append(byte)

    return bytes(byts)


def bit2hex(bits: Iterable[int]) -> str:
    """
    convert bits to hex string
    """
    return debitize(bits).hex()


def hex2bit(hex_str: str) -> list:
    """
    convert hex string to bits
    """
    return bitize(bytes.fromhex(hex_str))


def permute(raw_seq: Iterable, table: Iterable[int]) -> list:
    """
    permute bits with a table
    """
    # TODO: your code here
    # Use the permutation table to rearrange the bits
    permuted_seq = [raw_seq[i] for i in table]
    return permuted_seq  # just a placeholder


def xor(bits1: Iterable[int], bits2: Iterable[int]) -> 'list[int]':
    """
    xor two bits
    """
    # TODO: your code here
    # XOR bits1 and bits2 element-wise
    result = [bit1 ^ bit2 for bit1, bit2 in zip(bits1, bits2)]
    return result


class DES:
    # initial permutation
    IP = [
        57, 49, 41, 33, 25, 17, 9, 1,
        59, 51, 43, 35, 27, 19, 11, 3,
        61, 53, 45, 37, 29, 21, 13, 5,
        63, 55, 47, 39, 31, 23, 15, 7,
        56, 48, 40, 32, 24, 16, 8, 0,
        58, 50, 42, 34, 26, 18, 10, 2,
        60, 52, 44, 36, 28, 20, 12, 4,
        62, 54, 46, 38, 30, 22, 14, 6
    ]

    # final permutation
    FP = [
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25,
        32, 0, 40, 8, 48, 16, 56, 24
    ]

    # parity-bit drop table for key schedule
    KEY_DROP = [
        56, 48, 40, 32, 24, 16, 8, 0,
        57, 49, 41, 33, 25, 17, 9, 1,
        58, 50, 42, 34, 26, 18, 10, 2,
        59, 51, 43, 35, 62, 54, 46, 38,
        30, 22, 14, 6, 61, 53, 45, 37,
        29, 21, 13, 5, 60, 52, 44, 36,
        28, 20, 12, 4, 27, 19, 11, 3
    ]

    BIT_SHIFT = [
        1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1
    ]

    # key compression permutation
    KEY_COMPRESSION = [
        13, 16, 10, 23, 0, 4, 2, 27,
        14, 5, 20, 9, 22, 18, 11, 3,
        25, 7, 15, 6, 26, 19, 12, 1,
        40, 51, 30, 36, 46, 54, 29, 39,
        50, 44, 32, 47, 43, 48, 38, 55,
        33, 52, 45, 41, 49, 35, 28, 31
    ]

    # D box, key expansion permutation
    D_EXPANSION = [
        31, 0, 1, 2, 3, 4,
        3, 4, 5, 6, 7, 8,
        7, 8, 9, 10, 11, 12,
        11, 12, 13, 14, 15, 16,
        15, 16, 17, 18, 19, 20,
        19, 20, 21, 22, 23, 24,
        23, 24, 25, 26, 27, 28,
        27, 28, 29, 30, 31, 0
    ]

    # S boxes
    S1 = [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
    ]

    S2 = [
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
    ]

    S3 = [
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
    ]

    S4 = [
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
    ]

    S5 = [
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
    ]

    S6 = [
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
    ]

    S7 = [
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
    ]

    S8 = [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
    ]

    # S-box substitution
    S = [S1, S2, S3, S4, S5, S6, S7, S8]

    # D box, straight permutation
    D_STRAIGHT = [
        15, 6, 19, 20, 28, 11, 27, 16,
        0, 14, 22, 25, 4, 17, 30, 9,
        1, 7, 23, 13, 31, 26, 2, 8,
        18, 12, 29, 5, 21, 10, 3, 24
    ]

    @staticmethod
    def key_generation(key: 'list[int]') -> 'list[list[int]]':
        """
        raw_key: 64 bits
        return: 16 * (48bits key)
        """

        keys: 'list[list[int]]' = []
        # TODO: your code here
        # Permute the input key using the DES.KEY_DROP table
        key = permute(key, DES.KEY_DROP)
        # Split the permuted key into C and D parts
        C = key[:28]
        D = key[28:]
        sub_keys = []
        # Generate 16 subkeys
        for round_num in range(16):
            C = C[DES.BIT_SHIFT[round_num]:] + C[:DES.BIT_SHIFT[round_num]]
            D = D[DES.BIT_SHIFT[round_num]:] + D[:DES.BIT_SHIFT[round_num]]
            CD = C + D
            sub_key = permute(CD, DES.KEY_COMPRESSION)
            sub_keys.append(sub_key)

        return sub_keys

    @staticmethod
    def f(R: 'list[int]', key: 'list[int]') -> 'list[int]':
        """
        f function
        R: 32 bits
        key: 48 bits
        return: 32 bits
        """
        # TODO: your code here
        # Expansion: Expand R from 32 bits to 48 bits
        expanded_R = permute(R, DES.D_EXPANSION)

        # XOR the expanded R with the subkey
        xor_result = xor(expanded_R, key)

        # Apply S-box substitution
        s_box_output = []
        for i in range(8):
            # Take 6 bits from xor_result for each S-box
            s_box_input = xor_result[i * 6:(i + 1) * 6]
            # Extract row and column bits
            row = int(s_box_input[0] * 2 + s_box_input[5])
            col = int(''.join(map(str, s_box_input[1:5])), 2)
            # Retrieve the corresponding S-box value
            s_box_value = DES.S[i][row][col]
            # Convert the S-box value to binary (4 bits) and append to the result
            s_box_output.extend([int(bit) for bit in format(s_box_value, '04b')])

        # Permute the S-box output using the straight D-box
        result = permute(s_box_output, DES.D_STRAIGHT)

        return result

    @staticmethod
    def mixer(L: 'list[int]', R: 'list[int]', sub_key: 'list[int]') -> list:
        """
        right_half: 32 bits
        sub_key: 48 bits
        return: 32 bits
        """
        # TODO: your code here
        # Calculate the result of the f function
        f_result = DES.f(R, sub_key)

        # XOR the result of the f function with L
        new_R = xor(L, f_result)
        return new_R, R

    @staticmethod
    def swapper(L: 'list[int]', R: 'list[int]') -> 'tuple[list[int]]':
        """
        A free function for you, LMAO ^O^
        """
        return R, L

    def __init__(self, raw_key: bytes) -> None:
        # for encryption use
        self.keys = DES.key_generation(bitize(raw_key))

        # for decryption use
        self.reverse_keys = deepcopy(self.keys)
        self.reverse_keys.reverse()

    def enc_block(self, block: 'list[int]') -> 'list[int]':
        """
        Encrypt a block of 64 bits (8 bytes).
        block: 64 bits.
        return: 64 bits.
        """
        # TODO: your code here
        # Initial permutation
        block = permute(block, DES.IP)
        L, R = block[:32], block[32:]

        # Perform 16 rounds of encryption
        for i, key in enumerate(self.keys):
            L, R = DES.mixer(L, R, key)
            if i != len(self.keys) - 1:
                L, R = DES.swapper(L, R)

        # Final permutation
        result = permute(L + R, DES.FP)

        return result

    def dec_block(self, block: 'list[int]') -> 'list[int]':
        """
        similar to enc_block
        block: 64 bits
        return: 64 bits
        """
        # TODO: your code here
        # Initial permutation
        block = permute(block, DES.IP)
        L, R = block[:32], block[32:]

        # Perform 16 rounds of decryption
        for i, key in enumerate(self.reverse_keys):
            if i != 0:
                L, R = DES.swapper(L, R)
            L, R = DES.mixer(L, R, key)

        # Final permutation
        result = permute(L + R, DES.FP)

        return result

    def encrypt(self, msg_str: str) -> bytes:
        """
        Encrypt the whole message.
        Handle block division here.
        *Inputs are guaranteed to have a length divisible by 8.
        """
        # TODO: your code here
        # Padding: Ensure the message length is a multiple of 8 bytes
        length = len(msg_str)
        if length % 8 != 0:
            msg_str += '\0' * (8 - length % 8)
        msg_bytes = msg_str.encode('utf-8')

        # Create an empty list to store the encrypted blocks
        ciphertext_bits = []

        # Split the message into 8-byte blocks and perform encryption on each block
        for i in range(len(msg_bytes) // 8):
            block_bytes = msg_bytes[i * 8: (i + 1) * 8]
            block_bits = self.enc_block(bitize(block_bytes))
            ciphertext_bits.extend(block_bits)

        # Convert the bits of all blocks back into a byte string
        encrypted_message = debitize(ciphertext_bits)
        return encrypted_message

    def decrypt(self, msg_bytes: bytes) -> str:
        """
        Decrypt the whole message.
        Similar to encrypt.
        """
        # TODO: your code here
        # Create an empty list to store the decrypted blocks
        plaintext_bits = []

        # Split the encrypted message into 64-bit blocks and perform decryption on each block
        for i in range(len(msg_bytes) // 8):
            block_bytes = msg_bytes[i * 8: (i + 1) * 8]
            block_bits = self.dec_block(bitize(block_bytes))
            plaintext_bits.extend(block_bits)

        # Convert the bits of all blocks back into a byte string
        decrypted_message = debitize(plaintext_bits)

        return decrypted_message.decode('utf-8')
