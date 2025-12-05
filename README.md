# Expression Search

A high-performance expression search tool written in Rust that finds mathematical expressions producing a target sequence.

## Overview

Expression Search exhaustively generates and evaluates C-like expressions to find pairs of statements `(x = ...; y = ...)` that, when executed iteratively, produce a desired output sequence. The tool uses JIT compilation and multithreading to achieve high performance.

## Features

- **JIT Compilation** — Compiles expressions to native machine code for fast evaluation (supports x86_64 and aarch64)
- **Multithreaded Search** — Uses Rayon for parallel expression generation and testing
- **Semantic Deduplication** — Automatically prunes semantically equivalent expressions to reduce search space
- **Configurable Operators** — Supports arithmetic, bitwise, comparison, logical, and assignment operators
- **Length-based Search** — Generates expressions by character length for systematic exploration

## Supported Operators

| Category | Operators |
|----------|-----------|
| Arithmetic | `+` `-` `*` `/` `%` `**` |
| Bitwise | `\|` `^` `&` `~` `<<` `>>` |
| Comparison | `==` `!=` `<` `>` `<=` `>=` |
| Logical | `\|\|` `&&` `!` |
| Assignment | `=` `+=` `-=` `*=` `/=` `%=` `\|=` `^=` `&=` `<<=` `>>=` |
| Inc/Dec | `++x` `--x` `x++` `x--` |

## Usage

### Building

```bash
cargo build --release
```

### Running

```bash
cargo run --release
```

### Configuration

Edit `src/params.rs` to configure the search:

```rust
// Target sequence to find
pub const ANSWER: &[NumT] = &[1, 2, 2, 1, 1, 2, 1, 2, 2, 1, ...];

// Initial value ranges for variables
pub const INIT_X_MIN: NumT = -1;
pub const INIT_X_MAX: NumT = 1;
pub const INIT_Y_MIN: NumT = -1;
pub const INIT_Y_MAX: NumT = 1;

// Search parameters
pub const MAX_LENGTH: usize = 10;        // Maximum expression length
pub const MAX_CACHE_LENGTH: usize = 7;   // Length up to which expressions are cached

// Performance options
pub const USE_JIT: bool = true;          // Enable JIT compilation
pub const USE_MULTITHREAD: bool = true;  // Enable parallel search

// Expression options
pub const LITERALS: &[NumT] = &[1, 2, 3]; // Allowed literal values
pub const USE_PARENS: bool = true;        // Allow parentheses
pub const PRUNE_CONST_EXPR: bool = true;  // Skip constant-only expressions
```

You can also customize which operators are used by editing `BINARY_OPERATORS`, `UNARY_OPERATORS`, `ASSIGN_OPERATORS`, and `INCDEC_OPERATORS` arrays.

### Custom Matching Logic

Implement custom matching logic in the `Matcher` struct:

```rust
impl Matcher {
    pub fn match_one(&mut self, index: usize, output: NumT) -> bool {
        // Return true if output matches expected value at index
        1 - (output % 2) == ANSWER[index]
    }

    pub fn match_final(self, e_x: &Expr, e_y: &Expr) -> bool {
        // Additional validation after sequence matches
        true
    }
}
```

## Output

When a matching expression pair is found, it's printed in the format:

```
x=<init_x>, y=<init_y> : <statement_x>; <statement_y>
```

For example:
```
x=1, y=0 : x+=y+1; y=x-y
```

## Architecture

```
src/
├── main.rs    # Search algorithm and main loop
├── lib.rs     # Expression types, operators, evaluation, and printing
├── jit.rs     # JIT compiler for x86_64 and aarch64
└── params.rs  # Configuration parameters
```

### Search Strategy

1. **Phase 1 (Cached)**: Generate all expressions and statements up to `MAX_CACHE_LENGTH`, testing all combinations
2. **Phase 2 (DFS)**: For longer expressions, use depth-first search combining cached expressions with newly generated statements

### Expression Equivalence

Expressions are deduplicated based on semantic equivalence — two expressions are considered equal if they produce identical results for all combinations of input values in the range `[-4, 4]`.

## Dependencies

- **rayon** — Parallel iterators
- **hashbrown** — Fast hash maps with Rayon support
- **libc** — System calls for JIT memory management

## Performance Tips

1. Enable JIT compilation (`USE_JIT = true`) for ~10x faster evaluation
2. Use multithreading on multi-core systems (`USE_MULTITHREAD = true`)
3. Minimize the initial value ranges when possible
4. Prune unnecessary operators from the search space
5. Use smaller `MAX_CACHE_LENGTH` to reduce memory usage at the cost of search speed

## Acknowledgements

This project was influenced by [pysearch](https://github.com/lynn/pysearch) by Lynn — a brute-force search tool for short expressions used in code golf.

## License

MIT
