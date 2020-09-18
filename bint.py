"""
This module is very similar to bitstring (at PyPI):
https://github.com/scott-griffiths/bitstring
which is pretty awesome.

The difference is that this module focuses on integers and stores
an integer internally, which can save time on some operations.
Furthermore, the API is a bit more compact to aid simple conversions, such as:

>>> Bin(0x4142).bytes
b'AB'
>>> Bin(b'AB').int == 0x4142
True
>>> Bin(14, n=6).tuple
(0, 0, 1, 1, 1, 0)
>>> Bin(0x123, n=16).rol(8).hex
'2301'

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
    >>> Bin(b"AB")
    Bin(16706, n=16)

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
        """
        >>> Bin(1, 4).tuple
        (0, 0, 0, 1)
        >>> Bin(8, 4).tuple
        (1, 0, 0, 0)
        >>> Bin(16, 4).tuple
        Traceback (most recent call last):
        ValueError: integer out of range

        >>> Bin((1, 0, 0, 0)).int
        8
        >>> Bin((0, 0, 0, 1)).int
        1
        >>> Bin(Bin(123123, 32).tuple).int
        123123
        """
        if isinstance(spec, int) or type(spec).__name__ == "Integer":
            self.int = int(spec)
            self.n = n if n is not None else self.int.bit_length()
        elif isinstance(spec, Bin):
            self.int = spec.int
            self.n = n if n is not None else spec.n
        elif isinstance(spec, bytes):
            self.int = int.from_bytes(spec, "big")
            self.n = len(spec) * 8
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

    def empty(self):
        return self._new(0, 0)

    @classmethod
    def map_list(self, args, n):
        return [Bin(arg, n) for arg in args]

    def resize(self, n):
        """
        >>> Bin(3).resize(10)
        Bin(3, n=10)
        >>> Bin(3).resize(1)
        Traceback (most recent call last):
        ValueError: integer out of range
        """
        return Bin(self, n)

    def __index__(self):
        """
        Whenever Python needs to losslessly convert the numeric object
        to an integer object (such as in slicing, or in the built-in bin(),
        hex() and oct() functions).

        >>> [0, 10, 20, 30, 40][Bin(3)]
        30
        """
        return self.int

    def __int__(self):
        """
        >>> int(Bin(123))
        123
        """
        return self.int

    @classmethod
    def _new(cls, x, n):
        self = object.__new__(cls)
        self.int = int(x)
        self.n = n
        return self

    def _coerce_same_n(self, other):
        if not isinstance(other, Bin):
            return Bin(other, n=self.n)
        return other

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

    @property
    def bytes(self):
        r"""
        >>> Bin(0x4142, 24).bytes
        b'\x00AB'
        """
        return self.int.to_bytes((self.n + 7) // 8, "big")

    @property
    def hex(self):
        """
        >>> Bin(0xabc, 12).hex
        'abc'
        """
        return hex(self.int).zfill((self.n + 3) // 4).lstrip("0x")

    @property
    def bin(self):
        """
        >>> Bin(0xabc, 12).bin
        '101010111100'
        """
        return bin(self.int).zfill(self.n).lstrip("0b")

    def __eq__(self, other):
        other = self._coerce_same_n(other)
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
        """
        Hamming weight. Alias: hw

        >>> Bin(0).hamming()
        0
        >>> Bin(1).hamming()
        1
        >>> Bin(0xffffffff).hamming()
        32
        >>> Bin(2**64-1).hamming()
        64
        >>> Bin(int("10" * 999, 2)).hamming()
        999
        """
        return sum(self.tuple)
    hw = hamming

    def parity(self):
        """
        Parity of all bits.

        >>> Bin(0).parity()
        0
        >>> Bin(1).parity()
        1
        >>> Bin(4).parity()
        1
        >>> Bin(6).parity()
        0
        >>> Bin(2**100).parity()
        1
        >>> Bin(2**100 + 1).parity()
        0
        >>> Bin(2**100 ^ 7).parity()
        0
        >>> Bin(2**100 ^ 3).parity()
        1
        """
        return self.hw() & 1

    def __and__(self, other):
        other = Bin(other)
        n = max(self.n, other.n)
        return self._new(self.int & other.int, n=n)
    __rand__ = __and__

    def __xor__(self, other):
        other = Bin(other, n=self.n)
        n = max(self.n, other.n)
        return self._new(self.int ^ other.int, n=n)
    __rxor__ = __xor__

    def __or__(self, other):
        other = Bin(other, n=self.n)
        n = max(self.n, other.n)
        return self._new(self.int | other.int, n=n)
    __ror__ = __or__

    def __lshift__(self, n):
        y = (self.int << n) & self.mask
        return self._new(y, n=self.n)

    def __rshift__(self, n):
        y = (self.int >> n) & self.mask
        return self._new(y, n=self.n)

    def __invert__(self):
        y = self.int ^ self.mask
        return self._new(y, n=self.n)

    def scalar_bin(self, other):
        """
        Dot product in GF(2).

        >>> Bin(0).scalar_bin(0)
        0
        >>> Bin(1).scalar_bin(1)
        1
        >>> Bin(0xf731).scalar_bin(0xffffff)
        0
        >>> Bin(1).scalar_bin(3)
        1
        >>> Bin(7).scalar_bin(15)
        1

        Same as:

        >>> (Bin(7) @ 15) & 1
        1
        >>> (Bin(7) & 15).parity()
        1
        """
        return (self & other).parity()

    def scalar_int(self, other):
        """
        Dot product in integers. Aliased as overloaded @,
        similarly to .dot = @ in numpy.

        >>> Bin(0) @ 0
        0
        >>> Bin(1) @ 1
        1
        >>> Bin(0xf731) @ 0xffffff
        10
        >>> Bin(1) @ 3
        1
        >>> Bin(7) @ 15
        3

        Same as:

        >>> (Bin(7) & 15).hw()
        3
        """
        return (self & other).hw()

    # as .dot() = @ in numpy
    __matmul__ = __rmatmul__ = scalar_int

    def bit_product(self, mask):
        """
        Multiply bits selected by mask.
        (better method name?)

        >>> Bin("101").bit_product("101")
        1
        >>> Bin("101").bit_product("100")
        1
        >>> Bin("101").bit_product("001")
        1
        >>> Bin("101").bit_product("111")
        0

        >>> Bin("1111").bit_product("1111")
        1
        >>> Bin("1111").bit_product("1110")
        1
        >>> Bin("1111").bit_product("0111")
        1
        >>> Bin("0111").bit_product("1111")
        0
        >>> Bin("1110").bit_product("1111")
        0
        """
        mask = self._coerce_same_n(mask)
        return int(self.int & mask.int == mask.int)

    @classmethod
    def concat(cls, *args):
        """
        Concatenate bitstrings. Classmethod, varargs.

        >>> Bin.concat(Bin(128), Bin(255), Bin(1, n=8)).str
        '100000001111111100000001'
        """
        if not args:
            return Bin(0, n=0)
        ret = args[0]
        for arg in args[1:]:
            ret = ret._concat1(arg)
        return ret

    def _concat1(self, other):
        if not isinstance(other, Bin):
            raise TypeError(
                "Can not concatenate to non-Bin instances"
                "(can not determine width)"
            )
        y = (self.int << other.n) | other.int
        return self._new(y, n=self.n + other.n)

    def halves(self):
        """
        >>> Bin(0x79, 8).halves()
        (Bin(7, n=4), Bin(9, n=4))
        """
        return self.split(parts=2)

    def swap_halves(self):
        """
        >>> Bin(0x79, 8).swap_halves().hex
        '97'
        """
        l, r = self.halves()
        return Bin.concat(r, l)

    def split(self, parts=None, sizes=None):
        """
        Split the bitstring into several parts.
        Either:
        - into @parts same-sized chunks
        - into parts with sizes given by @sizes

        >>> Bin(0x123, 12).split(parts=3)
        (Bin(1, n=4), Bin(2, n=4), Bin(3, n=4))
        >>> Bin(0x9821, 16).split(sizes=(4, 4, 8))   # 0x21 == 33
        (Bin(9, n=4), Bin(8, n=4), Bin(33, n=8))
        """
        assert (parts is not None) ^ (sizes is not None)
        if parts:
            assert self.n % parts == 0
            ret = []
            n = self.n // parts
            mask = (1 << n) - 1
            x = self.int
            for i in range(parts):
                ret.append(self._new(x & mask, n=n))
                x >>= n
            return tuple(ret[::-1])
        if sizes:
            assert sum(sizes) == self.n
            ret = []
            x = self.int
            for n in reversed(sizes):
                mask = (1 << n) - 1
                ret.append(self._new(x & mask, n=n))
                x >>= n
            return tuple(ret[::-1])

    def __getitem__(self, idx):
        """
        >>> Bin(0x1234, 16)[0:4]
        Bin(1, n=4)
        >>> Bin(0x1234, 16)[4:12] # 0x23 == 35
        Bin(35, n=8)
        >>> Bin(0x1234, 16)[-4:]
        Bin(4, n=4)
        >>> Bin("101010")[::2]
        Bin(7, n=3)
        >>> Bin("101010")[0]
        1
        >>> Bin("101010")[1]
        0
        >>> Bin("101010")[2]
        1
        """
        if isinstance(idx, slice):
            # todo: optimize simple substring slices?
            # easy to mess up with out of bounds, negative indices, etc. ...
            return Bin(self.tuple[idx])
        else:
            idx = int(idx) % self.n
            return 1 & (self.int >> (self.n - 1 - idx))




def Bin8(x): return Bin(x, n=8)  # noqa
def Bin16(x): return Bin(x, n=16)  # noqa
def Bin32(x): return Bin(x, n=32)  # noqa
def Bin64(x): return Bin(x, n=64)  # noqa


def test_halves():
    """
    >>> Bin.concat(Bin(0x7, n=4), Bin(0xa, 4)).hex
    '7a'
    >>> Bin.concat(*Bin.map_list([1, 2, 3, 4, 5, 6, 10], 4)).hex
    '123456a'
    """
    pass
