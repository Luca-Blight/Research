def gcd(a, b):
    """Greatest Common Divisor"""
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


# Extended Euclidean Algorithm to find the modular multiplicative inverse.
def extended_euclidean(a, b):
    if a == 0:
        return b, 0, 1
    else:
        gcd, x1, y1 = extended_euclidean(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y


# To create a RSA key pair, first randomly pick the two prime numbers to obtain the maximum (max).
prime_one = 13
prime_two = 7
max_num = prime_one * prime_two  # Max number that can be encrypted

# Euler's totient function
phi_n = (prime_one - 1) * (prime_two - 1)

# Then pick a number to be the public key pub. As long as you know the two prime numbers, you can compute a corresponding private key priv from this public key.

# Public key
pub_key = 5

# Check if the public key is valid.
if gcd(pub_key, phi_n) == 1:
    # Use Extended Euclidean Algorithm to generate the private key.
    _, private_key, _ = extended_euclidean(pub_key, phi_n)

    # Sometimes the private key can be negative, so convert it to positive.
    private_key = private_key % phi_n

    print(f"Public key: {pub_key}")
    print(f"Private key: {private_key}")
else:
    print("Public key is not valid.")


def encrypt(number, pub_key, max_num):
    print(f"original number: {number}")

    encrypted_number = (number**pub_key) % max_num
    print(f"encrypted number is: {encrypted_number} ")
    return encrypted_number


def decrypt(encrypted_number, private_key, max_num):
    decrypted_number = (encrypted_number**private_key) % max_num
    print(f"decrypted number is: {decrypted_number}")
    return decrypted_number


encrypted_number = encrypt(4, pub_key, max_num)
descrypted_number = decrypt(encrypted_number, private_key, max_num)

text = "CLOUD"
numbers = [ord(char) for char in text]
print("UTF-8 representation of CLOUD:", numbers)
encrypted_message = [encrypt(num, pub_key, max_num) for num in numbers]
print("Encrypted Message:", encrypted_message)

decrypted_message = [decrypt(num, private_key, max_num) for num in numbers]
print("Decrypted Message:", decrypted_message)
