from time import time
import string
import random
import pyaes

def generate(length):
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for i in range(length))

def main(params):
    length = int(params['input1'])
    iteration = int(params['input2'])

    message = generate(length)
    # print("Message:")
    # print(message)

    # 128-bit key (16 bytes)
    KEY = b'\xa1\xf6%\x8c\x87}_\xcd\x89dHE8\xbf\xc9,'

    start = time()
    for loops in range(iteration):
        aes = pyaes.AESModeOfOperationCTR(KEY)
        ciphertext = aes.encrypt(message)
        # print("Encrypted:")
        # print(ciphertext)

        aes = pyaes.AESModeOfOperationCTR(KEY)
        plaintext = aes.decrypt(ciphertext)
        # print("Unecrypted")
        # print(plaintext)
        aes = None

    latency = time() - start
    ret_val = {}
    ret_val['latency'] = latency

    return ret_val