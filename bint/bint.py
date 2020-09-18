"""
This module is very similar to bitstring (at PyPI):
https://github.com/scott-griffiths/bitstring
which is pretty awesome.

The difference is that this module focuses on integers and stores
integer internally, which can save time on some operations.
Furthermore, the API is a bit more compact to aid simple conversions.

Currently, there is only Bin class for big endian integers
(most significant bits go first in bit strings).

If needed, something like BinLE will be defined later for little endians.
"""


class Bin:
    """
    Represents an integer in a fixed-width binary \
    and provides tools for working on binary representation.

    >>> Bin(16).n
    5
    >>> Bin(16, n=3)
    Traceback (most recent call last):
    ValueError: integer out of range

    >>> str(Bin(16))
    '10000'
    >>> str(Bin(16, 8))
    '00010000'

    >>> Bin(16).tuple
    (1, 0, 0, 0, 0)
    >>> Bin(16).list
    [1, 0, 0, 0, 0]
    >>> Bin(0x4142).bytes
    b'AB'
    >>> Bin("1101").int
    13
    >>> Bin("1110001101").hex
    '38d'
    >>> str(Bin(16))
    '10000'
    >>> repr(Bin(16))
    'Bin(16, n=5)'

    >>> Bin("1101") == Bin((1, 1, 0, 1)) == Bin([1, 1, 0, 1]) == Bin(13)
    True
    """
    __slots__ = "int", "n"

    def __init__(self, spec, n=None):
        if isinstance(spec, int) or type(spec).__name__ == "Integer":
            self.int = int(spec)
            self.n = n if n is not None else self.int.bit_length()
        elif isinstance(spec, Bin):
            self.int = spec.int
            self.n = n if n is not None else spec.n
        elif isinstance(spec, bytes):
            self.int = int.from_bytes(spec, "big")
        else:
            # vector / tuple / list
            assert n is None or n == len(spec)
            self.n = len(spec)
            self.int = sum(
                int(b) << (self.n - 1 - i)
                for i, b in enumerate(spec)
            )
        if not (0 <= self.int < (1 << self.n)):
            raise ValueError("integer out of range")

    def resize(self, n):
        return Bin(self, n)

    @classmethod
    def _new(cls, x, n):
        self = object.__new__(cls)
        self.int = int(x)
        self.n = n
        return self

    def __int__(self):
        return self.int

    @property
    def tuple(self):
        return tuple(map(int, bin(self.int).lstrip("0b").zfill(self.n)))

    @property
    def list(self):
        return list(map(int, bin(self.int).lstrip("0b").zfill(self.n)))

    def __iter__(self):
        return iter(self.tuple)

    def __str__(self):
        return "".join("%d" % v for v in self.tuple)
    str = property(__str__)

    def __repr__(self):
        return "Bin(%d, n=%d)" % (self.int, self.n)

    def __bytes__(self):
        return self.int.to_bytes((self.n + 7) // 8, "big")
    bytes = property(__bytes__)

    def __hex__(self):
        return hex(self.int).lstrip("0x")
    hex = property(__hex__)

    def __eq__(self, other):
        if not isinstance(other, Bin):
            other = Bin(other, n=self.n)
        if other.n != self.n:
            raise ValueError("Can not compare Bin's with different n")
        return self.int == other.int

    @property
    def mask(self):
        return (1 << (self.n)) - 1

    def rol(self, n):
        """
        Rotate left by @n bits

        >>> hex( Bin(0x1234, 16).rol(4).int )
        '0x2341'
        >>> hex( Bin(0x1234, 16).rol(12).int )
        '0x4123'
        """

        n %= self.n
        x = self.int
        y = ((x << n) | (x >> (self.n - n))) & self.mask
        return self._new(y, n=n)

    def ror(self, n):
        """
        Rotate right by @n bits

        >>> hex( Bin(0x1234, 16).ror(4).int )
        '0x4123'
        >>> hex( Bin(0x1234, 16).ror(12).int )
        '0x2341'
        """
        return self.rol(-n)

    def hamming(self):
        return sum(self.tuple)
    hw = hamming

    def parity(self):
        return self.hw() & 1

    def __and__(self, other):
        other = Bin(other)
        n = max(self.n, other.n)
        return self._new(self.int & other.int, n=n)

    def __xor__(self, other):
        other = Bin(other, n=self.n)
        n = max(self.n, other.n)
        return self._new(self.int ^ other.int, n=n)

    def __or__(self, other):
        other = Bin(other, n=self.n)
        n = max(self.n, other.n)
        return self._new(self.int | other.int, n=n)

    def __lshift__(self, n):
        y = (self.int << n) & self.mask
        return self._new(y, n=self.n)

    def __rshift__(self, n):
        y = (self.int >> n) & self.mask
        return self._new(y, n=self.n)

    def __invert__(self):
        y = self.int ^ self.mask
        return self._new(y, n=self.n)

    def bit_product(self, mask):
        return int(self.x & mask == mask)


def Bin8(x): return Bin(x, n=8)  # noqa
def Bin16(x): return Bin(x, n=16)  # noqa
def Bin32(x): return Bin(x, n=32)  # noqa
def Bin64(x): return Bin(x, n=64)  # noqa
