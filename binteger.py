"""
`binteger` is a toolkit for manipulating integers in their binary
(fixed-width) representation (big endian - most significant bits go first).

**Documentation**: https://binteger.readthedocs.io/

**Code**: https://github.com/hellman/binteger

**Installation**::

    pip install binteger

This module is quite similar to `bitstring` (at PyPI):
https://github.com/scott-griffiths/bitstring
which is pretty awesome.

The difference is that the API is a bit more compact
to aid simple conversions, such as:

>>> from binteger import Bin
>>> Bin(0x4142).bytes
b'AB'
>>> Bin(b'AB').int == 0x4142
True
>>> Bin(14, n=6).tuple
(0, 0, 1, 1, 1, 0)
>>> Bin(0x123, n=16).rol(8).hex
'2301'

.. warning:: `binteger` is optimized for *convenience of use*,
              rather than for performance.
"""
from pprint import pprint
from random import randrange
from functools import reduce
from itertools import product


class Bin:
    """An integer with a fixed-width binary representation.

    Provides tools for working on binary representation.

    Parameters
    ----------
    spec:
        Specifier for the integer:
        int,
        or list/tuple/str of 0/1 (binary representation),
        or bytes (to be unpacked).
    n: int, optional
        Width of the binary representation.

    Attributes
    ----------
    int: int
        Integer representation.
    n: int
        Width of the binary representation.

    Examples
    --------
    Creation:

    >>> Bin(16).n
    5
    >>> Bin(16).int
    16
    >>> Bin(16, n=3)
    Traceback (most recent call last):
    ValueError: integer out of range

    >>> Bin(b"AB")  # bytes are unpacked
    Bin(0b0100000101000010, n=16)
    >>> Bin("010")  # strings must be the binary representation
    Bin(0b010, n=3)
    >>> Bin("AB")  # strings must be 01-formed
    Traceback (most recent call last):
    ValueError: invalid literal for int() with base 10: 'A'

    Outputting (formatting):

    >>> str(Bin(16))
    '10000'
    >>> repr(Bin(16))
    'Bin(0b10000, n=5)'
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
    >>> Bin("1101") == Bin((1, 1, 0, 1)) == Bin([1, 1, 0, 1]) == Bin(13)
    True

    >>> Bin([0, 1, 2])
    Traceback (most recent call last):
    ValueError: integer 2 is not binary

    >>> Bin([]).tuple
    ()
    >>> Bin(0).tuple
    ()
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
        >>> Bin({1, 4, 5, 6}, 8)
        Bin(0b01001110, n=8)
        """
        # support Sage integers (ZZ) and IntegerMod etc.
        if isinstance(spec, int) or type(spec).__name__.startswith("Integer"):
            self.int = int(spec)
            self.n = n if n is not None else self.int.bit_length()
        elif isinstance(spec, Bin):
            self.int = spec.int
            self.n = n if n is not None else spec.n
        elif isinstance(spec, bytes):
            self.int = int.from_bytes(spec, "big")
            self.n = len(spec) * 8
        elif isinstance(spec, (set, frozenset)):
            if n is None:
                raise ValueError("n must be specified for set-of-indexes spec")
            self.int = sum(2**(n-1-i) for i in spec)
            self.n = n if n is not None else self.int.bit_length()
        else:
            # vector / tuple / list
            assert n is None or n == len(spec)
            self.n = len(spec)
            self.int = sum(
                _int01(b) << (self.n - 1 - i)
                for i, b in enumerate(spec)
            )
        if not (0 <= self.int < (1 << self.n)):
            raise ValueError("integer out of range")

    @classmethod
    def empty(self, n=0):
        """Create empty :class:`Bin`.

        >>> Bin.empty(0)
        Bin(0b0, n=0)
        >>> Bin.empty(4)
        Bin(0b0000, n=4)
        >>> Bin.zero(4)
        Bin(0b0000, n=4)
        """
        n = _check_n(n)
        return self._new(0, n)
    zero = empty

    @classmethod
    def full(self, n):
        """Create full :class:`Bin` of width `n`.

        >>> Bin.full(0)
        Bin(0b0, n=0)
        >>> Bin.full(4)
        Bin(0b1111, n=4)
        >>> Bin.ones(4)
        Bin(0b1111, n=4)
        """
        n = _check_n(n)
        return self._new(2**n-1, n)
    ones = full

    @classmethod
    def iter(self, *ns):
        """Iterate over all `n`-bit :class:`Bin`s.
        Multiple arguments yield product of single iters,
        i.e., all tuples of :class:`Bin`s of given sizes `ns`.

        >>> list(Bin.iter(2))
        [Bin(0b00, n=2), Bin(0b01, n=2), Bin(0b10, n=2), Bin(0b11, n=2)]
        >>> pprint(list(Bin.iter(1, 2)))
        [(Bin(0b0, n=1), Bin(0b00, n=2)),
         (Bin(0b0, n=1), Bin(0b01, n=2)),
         (Bin(0b0, n=1), Bin(0b10, n=2)),
         (Bin(0b0, n=1), Bin(0b11, n=2)),
         (Bin(0b1, n=1), Bin(0b00, n=2)),
         (Bin(0b1, n=1), Bin(0b01, n=2)),
         (Bin(0b1, n=1), Bin(0b10, n=2)),
         (Bin(0b1, n=1), Bin(0b11, n=2))]
        """
        if len(ns) == 1:
            n = ns[0]
            n = _check_n(n)
            for x in range(2**n):
                yield Bin(x, n)
        else:
            yield from product(*[self.iter(n) for n in ns])

    @classmethod
    def unit(cls, i, n):
        r""":class:`Bin` integer only with bit at index `i` set
        (equals :math:`2^{n-1-(i\mod n)}`)

        Parameters
        ----------
        i : int
            The index, requires :math:`-n \le i < n`.
        n : int
            The width.

        Examples
        --------
        >>> Bin.unit(0, 5)
        Bin(0b10000, n=5)
        >>> Bin.unit(3, 5)
        Bin(0b00010, n=5)
        >>> Bin.unit(4, 5)
        Bin(0b00001, n=5)
        >>> Bin.unit(-5, 5)
        Bin(0b10000, n=5)
        >>> Bin.unit(-1, 5)
        Bin(0b00001, n=5)
        >>> Bin.unit(5, 5)
        Traceback (most recent call last):
        ValueError: integer out of range
        >>> Bin.unit(-6, 5)
        Traceback (most recent call last):
        ValueError: integer out of range
        """
        if not (-n <= i < n):
            raise ValueError("integer out of range")
        return cls(1 << (n - 1 - i % n), n)

    @classmethod
    def random(cls, n, nonzero=False):
        r"""Random `n`-bit Bin.

        Parameters
        ----------
        n : int
            The width.
        nonzero : bool = False
            Force nonzero integer?

        Examples
        --------
        >>> Bin.random(0)
        Bin(0b0, n=0)
        >>> Bin.random(10)  # doctest: +SKIP
        Bin(0b0110110011, n=10)
        >>> Bin.random(10)  # doctest: +SKIP
        Bin(0b1010000111, n=10)
        >>> all(Bin.random(3) for _ in range(1000))
        False
        >>> all(Bin.random(3, nonzero=True) for _ in range(1000))
        True
        """
        return cls(randrange(1 if nonzero else 0, 2**n), n)

    @property
    def mask(self):
        """Bin: Binary mask for the full width."""
        return self._new((1 << (self.n)) - 1, self.n)

    # =========================================================
    # Creation
    # =========================================================
    @classmethod
    def _new(cls, x, n):
        self = object.__new__(cls)
        self.int = int(x)
        self.n = n
        return self

    def copy(self):
        return self._new(self.int, self.n)

    @classmethod
    def _coerce_hint_n(self, other, n):
        if not isinstance(other, Bin):
            return Bin(other, n=n)
        return other

    @classmethod
    def _coerce_force_n(self, other, n):
        if not isinstance(other, Bin):
            return Bin(other, n=n)
        return other.resize(n)

    def _coerce_hint_same_n(self, other):
        if not isinstance(other, Bin):
            return Bin(other, n=self.n)
        return other

    def _coerce_same_n(self, other):
        if not isinstance(other, Bin):
            return Bin(other, n=self.n)
        return other.resize(self.n)

    @classmethod
    def array(self, *args, n=None, ns=None):
        """Convert a list of objects to a list of :class:`Bin`
        with the same `n`, or an iterable `ns` of sizes.

        >>> Bin.array(1, 2, 3, 4, n=4)
        [Bin(0b0001, n=4), Bin(0b0010, n=4), Bin(0b0011, n=4), Bin(0b0100, n=4)]
        >>> Bin.array(1, 2, 3, 4, ns=(1,2,3,4))
        [Bin(0b1, n=1), Bin(0b10, n=2), Bin(0b011, n=3), Bin(0b0100, n=4)]
        >>> Bin.concat(*Bin.array(1, 2, 3, 4, n=4)).hex
        '1234'
        >>> Bin.concat(*Bin.array(1, 2, 3, 4, ns=(4, 8, 4, 16))).hex
        '10230004'
        """
        assert (n is None) or (ns is None)
        if n:
            return [Bin(arg, n) for arg in args]
        if ns:
            return [Bin(arg, n) for arg, n in zip(args, ns)]
        return [Bin(arg) for arg in args]

    def resize(self, n, truncate=False):
        """Change the width of the integer
        (`truncate` must be set to True to force truncation,
        otherwise an Exceptino will be raised if doesn't fit).

        Examples
        --------
        >>> Bin(3).resize(10)
        Bin(0b0000000011, n=10)
        >>> Bin(3).resize(1)
        Traceback (most recent call last):
        ValueError: integer out of range
        >>> Bin(3).resize(1, truncate=True)
        Bin(0b1, n=1)
        """
        n = _check_n(n)
        if truncate:
            mask = (1 << n) - 1
            return Bin(self.int & mask, n)
        else:
            return Bin(self.int, n)

    # =========================================================
    # Output something
    # =========================================================
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

    def __len__(self):
        return self.n

    @property
    def tuple(self):
        """Tuple of the binary representation.

        Examples
        --------
        >>> Bin(0x123).tuple
        (1, 0, 0, 1, 0, 0, 0, 1, 1)
        >>> Bin(0x123, 10).tuple
        (0, 1, 0, 0, 1, 0, 0, 0, 1, 1)
        """
        return tuple(map(int, self.bin))

    @property
    def list(self):
        """List of the binary representation.

        Examples
        --------
        >>> Bin(0x123).list
        [1, 0, 0, 1, 0, 0, 0, 1, 1]
        >>> Bin(0x123, 10).list
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 1]
        """
        return list(map(int, self.bin))

    @property
    def vector(self):
        """GF(2)-vector of the binary representation.

        .. warning: Only in sage mode. TBD: better imports
        """
        from sage.all import vector, GF
        return vector(GF(2), self.tuple)

    @property
    def support(self):
        """Tuple of indices `i` such that bits with index `i` is set.

        >>> Bin(0x1234, 16).support
        (3, 6, 10, 11, 13)
        >>> Bin(0x1234).support
        (0, 3, 7, 8, 10)
        """
        return tuple(i for i, c in enumerate(self.tuple) if c)

    def __iter__(self):
        return iter(self.tuple)

    def __str__(self):
        """01-string binary representation."""
        return "".join("%d" % v for v in self.tuple)
    str = property(__str__)

    def __repr__(self):
        s = str(self) if self.n else "0"
        return f"Bin(0b{s}, n={self.n})"

    @property
    def bytes(self):
        r"""
        Bytes representation (big-endian).

        >>> Bin(0x4142, 24).bytes
        b'\x00AB'
        """
        return self.int.to_bytes((self.n + 7) // 8, "big")

    @property
    def hex(self):
        """Hexadecimal representation.

        >>> Bin(0xabc, 12).hex
        'abc'
        >>> Bin(0xabc, 13).hex
        '0abc'
        >>> Bin(0xabc, 16).hex
        '0abc'
        """
        return f"{self.int:x}".zfill((self.n + 3) // 4)

    @property
    def bin(self):
        """
        >>> Bin(0xabc, 12).bin
        '101010111100'
        >>> Bin(0xabc, 13).bin
        '0101010111100'
        >>> Bin(0xabc, 16).bin
        '0000101010111100'
        """
        if not self.n:
            return ""
        return f"{self.int:b}".zfill(self.n)

    def __getitem__(self, idx):
        """
        >>> Bin(0x1234, 16)[0:4]
        Bin(0b0001, n=4)
        >>> Bin(0x1234, 16)[4:12] # 0x23 == 35
        Bin(0b00100011, n=8)
        >>> Bin(0x1234, 16)[-4:]
        Bin(0b0100, n=4)
        >>> Bin("101010")[::2]
        Bin(0b111, n=3)
        >>> Bin("101010")[0]
        1
        >>> Bin("101010")[1]
        0
        >>> Bin("101010")[2]
        1
        >>> Bin("101010")[5]
        0
        >>> Bin("101010")[6]
        Traceback (most recent call last):
        IndexError: index 6 out of range (-6,5)
        >>> Bin("101010")[-6]
        1
        >>> Bin("101010")[-7]
        Traceback (most recent call last):
        IndexError: index -7 out of range (-6,5)
        """
        if isinstance(idx, slice):
            # todo: optimize simple substring slices?
            # easy to mess up with out of bounds, negative indices, etc. ...
            return Bin(self.tuple[idx])
        else:
            idx = int(idx)
            if idx < -self.n or idx >= self.n:
                raise IndexError(f"index {idx} out of range ({-self.n},{self.n-1})")
            if idx < 0:
                idx += self.n
            assert 0 <= idx < self.n
            return 1 & (self.int >> (self.n - 1 - idx))

    # __setitem__ maybe ?
    # do we want to keep this immutable?

    # =========================================================
    # Comparison
    # =========================================================
    def __eq__(self, other):
        other = self._coerce_hint_same_n(other)
        if other.n != self.n:
            raise ValueError("Can not compare Bin's with different n")
        return self.int == other.int

    def __lt__(self, other):
        """
        >>> Bin(3, 5) < Bin(4, 5)
        True
        >>> Bin(3, 5) < Bin(3, 5)
        False
        >>> Bin(4, 5) < Bin(3, 5)
        False
        >>> Bin(3, 5) < 4
        True
        >>> Bin(3, 5) < 3
        False
        >>> Bin(4, 5) < 3
        False
        """
        other = self._coerce_same_n(other)
        return self.int < other.int

    def __le__(self, other):
        other = self._coerce_same_n(other)
        return self.int <= other.int

    def __gt__(self, other):
        other = self._coerce_same_n(other)
        return self.int > other.int

    def __ge__(self, other):
        other = self._coerce_same_n(other)
        return self.int >= other.int

    def __bool__(self):
        """
        >>> bool(Bin("000"))
        False
        >>> bool(Bin("100"))
        True
        >>> bool(Bin("101") & 1)
        True
        >>> bool(Bin("100") & 1)
        False
        """
        return self.int != 0

    def __hash__(self):
        """
        Hash matches that of underlying int to allow reusing Bin as dict keys

        >>> d = {10: 123}; d[Bin(10, 5)] == d[Bin(10, 100)] == 123
        True
        >>> hash(100) == hash(Bin(100, 7)) == hash(Bin(100, 10))
        True
        """
        return hash(self.int)

    def is_prec(self, other):
        return self.int & other == self.int and self.int != other

    def is_preceq(self, other):
        return self.int & other == self.int

    def is_succ(self, other):
        return self.int & other == other and self.int != other

    def is_succeq(self, other):
        return self.int & other == other

    # =========================================================
    # Properties
    # =========================================================
    @property
    def weight(self):
        """Hamming weight.
        (Aliases: :attr:`hw`, :attr:`wt`).

        >>> Bin(0).weight
        0
        >>> Bin(1).weight
        1
        >>> Bin(0xffffffff).weight
        32
        >>> Bin(2**64-1).weight
        64
        >>> Bin(int("10" * 999, 2)).weight
        999
        """
        return sum(self.tuple)
    wt = hw = weight

    @property
    def parity(self):
        """
        Parity of all bits.

        >>> Bin(0).parity
        0
        >>> Bin(1).parity
        1
        >>> Bin(4).parity
        1
        >>> Bin(6).parity
        0
        >>> Bin(2**100).parity
        1
        >>> Bin(2**100 + 1).parity
        0
        >>> Bin(2**100 ^ 7).parity
        0
        >>> Bin(2**100 ^ 3).parity
        1
        """
        return self.weight & 1

    # =========================================================
    # Transformations / operations
    # =========================================================
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
        return self._new(y, n=self.n)

    def ror(self, n):
        """
        Rotate right by @n bits

        >>> hex( Bin(0x1234, 16).ror(4).int )
        '0x4123'
        >>> hex( Bin(0x1234, 16).ror(12).int )
        '0x2341'
        >>> Bin(1, 16).rol(1).n
        16
        """
        return self.rol(-n)

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

    def __neg__(self):
        """
        >>> -Bin(1, 4)
        Bin(0b1111, n=4)
        >>> -Bin(0, 4)
        Bin(0b0000, n=4)
        """
        mod = 1 << self.n
        v = mod - self.int
        if v >= mod:
            v -= mod
        return self._new(v, n=self.n)

    def __add__(self, other):
        """
        >>> Bin(31, 5) + 2
        Bin(0b00001, n=5)
        >>> Bin(31, 5) + Bin(1, 5)
        Bin(0b00000, n=5)
        >>> Bin(31, 5) + Bin(1, 6)
        Traceback (most recent call last):
         ...
        ValueError: add/sub Bin with different n? 5 vs 6
        """
        if isinstance(other, Bin):
            if self.n != other.n:
                raise ValueError(f"add/sub Bin with different n? {self.n} vs {other.n}")
            other = other.int
        mask = (1 << self.n) - 1
        v = int(self.int + other) & mask
        return self._new(v, n=self.n)

    __radd__ = __add__

    def __sub__(self, other):
        """
        >>> Bin(1, 5) - 2
        Bin(0b11111, n=5)
        >>> Bin(17, 5) - Bin(1, 5)
        Bin(0b10000, n=5)
        >>> Bin(1, 5) + Bin(31, 6)
        Traceback (most recent call last):
         ...
        ValueError: add/sub Bin with different n? 5 vs 6
        """
        if isinstance(other, Bin):
            if self.n != other.n:
                raise ValueError(f"add/sub Bin with different n? {self.n} vs {other.n}")
            other = other.int
        mask = (1 << self.n) - 1
        v = int(self.int - other) & mask
        return self._new(v, n=self.n)

    def __rsub__(self, other):
        """
        >>> Bin(1, 5) - 2
        Bin(0b11111, n=5)
        >>> Bin(17, 5) - Bin(1, 5)
        Bin(0b10000, n=5)
        >>> Bin(1, 5) + Bin(31, 6)
        Traceback (most recent call last):
         ...
        ValueError: add/sub Bin with different n? 5 vs 6
        """
        if isinstance(other, Bin):
            if self.n != other.n:
                raise ValueError(f"add/sub Bin with different n? {self.n} vs {other.n}")
            other = other.int
        mask = (1 << self.n) - 1
        v = int(other - self.int) & mask
        return self._new(v, n=self.n)

    def scalar_bin(self, other):
        """Dot product in GF(2).

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
        >>> (Bin(7) & 15).parity
        1
        """
        return (self & other).parity

    def scalar_int(self, other):
        """Dot product in integers. Aliased as overloaded @,
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

        >>> (Bin(7) & 15).weight
        3
        """
        return (self & other).weight

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
    def concat(cls, *args, n=None, ns=None):
        """
        Concatenate bitstrings. Classmethod, varargs.
        :param:

        >>> Bin.concat(Bin(128), Bin(255), Bin(1, n=8)).str
        '100000001111111100000001'
        >>> Bin.concat(1, 2, 3, 4).str
        '11011100'
        >>> Bin.concat(1, 2, 3, 4, n=3).str
        '001010011100'
        >>> Bin.concat(1, 2, 3, 15, n=4).str
        '0001001000111111'
        >>> Bin.concat(1, 2, 3, 16, n=4).str
        Traceback (most recent call last):
        ValueError: integer out of range
        """
        if not args:
            return Bin(0, n=0)
        args = cls.array(*args, n=n, ns=ns)
        return reduce(lambda a, b: a._concat1(b), args)

    def _concat1(self, other, n=None):
        y = (self.int << other.n) | other.int
        return self._new(y, n=self.n + other.n)

    def halves(self):
        """
        >>> Bin(0x79, 8).halves()
        (Bin(0b0111, n=4), Bin(0b1001, n=4))
        """
        return self.split(parts=2)

    def swap_halves(self):
        """
        >>> Bin(0x79, 8).swap_halves().hex
        '97'
        """
        l, r = self.halves()
        return Bin.concat(r, l)

    def split(self, parts=None, n=None, ns=None):
        """
        Split the bitstring into several parts.
        Either:
        - into @parts same-sized chunks
        - into parts with sizes @n each or given by @ns

        >>> Bin(0x123, 12).split(3) == Bin(0x123, 12).split(parts=3)
        True
        >>> Bin(0x123, 12).split(parts=3)
        (Bin(0b0001, n=4), Bin(0b0010, n=4), Bin(0b0011, n=4))
        >>> Bin(0x9821, 16).split(ns=(4, 4, 8))   # 0x21 == 33
        (Bin(0b1001, n=4), Bin(0b1000, n=4), Bin(0b00100001, n=8))
        >>> Bin(0x9821, 16).split(n=4)
        (Bin(0b1001, n=4), Bin(0b1000, n=4), Bin(0b0010, n=4), Bin(0b0001, n=4))
        """
        assert 1 == (parts is not None) + (ns is not None) + (n is not None)
        if parts or n:
            if parts:
                assert self.n % parts == 0
                n = self.n // parts
            else:
                assert self.n % n == 0
                parts = self.n // n

            ret = []
            mask = (1 << n) - 1
            x = self.int
            for i in range(parts):
                ret.append(self._new(x & mask, n=n))
                x >>= n
            return tuple(ret[::-1])
        if ns:
            assert sum(ns) == self.n
            ret = []
            x = self.int
            for n in reversed(ns):
                mask = (1 << n) - 1
                ret.append(self._new(x & mask, n=n))
                x >>= n
            return tuple(ret[::-1])

    def squeeze_by_mask(self, mask):
        """
        Keep bits selected by mask and delete the others.

        Parameters
        ----------
        mask: :class:`Bin` or int or str
            Mask for squeeze, will be converted to ``Bin(..., self.n)``

        Examples
        --------
        >>> Bin("10101").squeeze_by_mask("10101").str
        '111'
        >>> Bin("10101").squeeze_by_mask("01111").str
        '0101'
        """
        mask = self._coerce_same_n(mask).int
        res = 0
        n = 0
        x = self.int
        while mask:
            if mask & 1:
                res |= (x & 1) << n
                n += 1
            mask >>= 1
            x >>= 1
        return self._new(res, n=n)


def Bin8(x):
    """Fixed-width 8-bit :class:`Bin` (shortcut to ``Bin(x, 8)``)   ."""
    return Bin(x, n=8)

def Bin16(x):
    """Fixed-width 16-bit :class:`Bin` (shortcut to ``Bin(x, 16)``) ."""
    return Bin(x, n=16)

def Bin32(x):
    """Fixed-width 32-bit :class:`Bin` (shortcut to ``Bin(x, 32)``) ."""
    return Bin(x, n=32)

def Bin64(x):
    """Fixed-width 64-bit :class:`Bin` (shortcut to ``Bin(x, 64)``) ."""
    return Bin(x, n=64)



def _int01(v):
    w = int(v)
    if not 0 <= w <= 1:
        raise ValueError("integer %d is not binary" % w)
    return w


def _check_n(n):
    n = int(n)
    if n < 0:
        raise ValueError("Negative width?")
    return n


def _test_halves():
    """
    >>> Bin.concat(Bin(0x7, n=4), Bin(0xa, 4)).hex
    '7a'
    >>> Bin.concat(*Bin.array(1, 2, 3, 4, 5, 6, 10, n=4)).hex
    '123456a'
    >>> Bin.concat(1, 2, 3, 4, 5, 6, 10, n=4).hex
    '123456a'
    """
    pass
