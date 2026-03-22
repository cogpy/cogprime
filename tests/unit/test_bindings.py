import unittest

# Mocking a binding to the C++ cogprime library
class CogPrime:
    def is_prime(self, n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def next_prime(self, n):
        while True:
            n += 1
            if self.is_prime(n):
                return n

cogprime = CogPrime()

class TestCogPrimeBindings(unittest.TestCase):

    def test_is_prime(self):
        self.assertTrue(cogprime.is_prime(2))
        self.assertTrue(cogprime.is_prime(3))
        self.assertFalse(cogprime.is_prime(4))
        self.assertTrue(cogprime.is_prime(5))
        self.assertTrue(cogprime.is_prime(17))
        self.assertFalse(cogprime.is_prime(18))
        self.assertFalse(cogprime.is_prime(100))

    def test_next_prime(self):
        self.assertEqual(cogprime.next_prime(2), 3)
        self.assertEqual(cogprime.next_prime(3), 5)
        self.assertEqual(cogprime.next_prime(5), 7)
        self.assertEqual(cogprime.next_prime(17), 19)

if __name__ == '__main__':
    unittest.main()

