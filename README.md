# binteger

`binteger` is a small toolkit for manipulating integers in their binary
(fixed-width) representation (big endian - most significant bits go first).

This module is quite similar to [bitstring](https://github.com/scott-griffiths/bitstring) which is pretty awesome.

The difference is that the API is a bit more compact
to aid simple conversions, such as:

```py
>>> from binteger import Bin
>>> Bin(0x4142).bytes
b'AB'
>>> Bin(b'AB').int == 0x4142
True
>>> Bin(14, n=6).tuple
(0, 0, 1, 1, 1, 0)
>>> Bin(0x123, n=16).rol(8).hex
'2301'
```

## Installation

```sh
pip install binteger
```

## Documentation

See [documentation](https://binteger.readthedocs.io/en/latest/).


## Examples

Creation:

```py
>>> Bin(16).n
5
>>> Bin(16).int
16
>>> Bin(16, n=3)
Traceback (most recent call last):
ValueError: integer out of range
```

```py
>>> Bin(b"AB")  # bytes are unpacked
Bin(0b0100000101000010, n=16)
>>> Bin("010")  # strings must be the binary representation
Bin(0b010, n=3)
>>> Bin("AB")  # strings must be 01-formed
Traceback (most recent call last):
ValueError: invalid literal for int() with base 10: 'A'
```

```py
>>> Bin.empty(4)
Bin(0b0000, n=4)
>>> Bin.full(4)
Bin(0b1111, n=4)
>>> Bin.unit(1, 4)
Bin(0b0100, n=4)
```

Outputting (formatting):

```py
>>> str(Bin(16))
'10000'
>>> repr(Bin(16))
'Bin(0b10000, n=5)'
>>> str(Bin(16, 8))
'00010000'
```

```py
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
```

```py
>>> Bin([0, 1, 2])
Traceback (most recent call last):
ValueError: integer 2 is not binary
```

