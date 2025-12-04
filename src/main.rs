//! Expression search - finds expressions that produce a target sequence.
//!
//! Configure the search by editing `params.rs`.

mod params;

use expr_rust::{Expr, Operator, print_expression, NumT, naive_eval, jit};
use hashbrown::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use rayon::prelude::*;

use params::*;

type Cache = HashSet<Arc<Expr>>;

struct Context {
    expressions: HashMap<usize, Cache>,
    statements: HashMap<usize, Cache>,
    var_expressions: Vec<Arc<Expr>>,
}

impl Context {
    fn new() -> Self {
        Self {
            expressions: HashMap::new(),
            statements: HashMap::new(),
            var_expressions: Vec::new(),
        }
    }
}

// =============================================================================
// EXPRESSION CONSTRUCTION
// =============================================================================

fn make_expr(left: Option<Arc<Expr>>, right: Option<Arc<Expr>>, literal: NumT, op: Operator) -> Expr {
    let mut e = Expr { left, right, literal, op, jit: None };
    if USE_JIT {
        let jit = jit::jit_compile_expr(&e);
        e.jit = Some(Arc::new(jit));
    }
    e
}

fn cache_expression(cache: &mut Cache, e: Expr) {
    cache.insert(Arc::new(e));
}

fn make_statement(var: &Arc<Expr>, expr: &Arc<Expr>, op: Operator) -> Expr {
    make_expr(Some(var.clone()), Some(expr.clone()), -1, op)
}

// =============================================================================
// EVALUATION
// =============================================================================

#[inline]
fn eval_naive(e_x: &Expr, e_y: &Expr, x: &mut NumT, y: &mut NumT) -> bool {
    let mut fatal = false;
    naive_eval(e_x, x, y, &mut fatal);
    if fatal { return false; }
    naive_eval(e_y, y, x, &mut fatal);
    !fatal
}

#[inline]
fn eval_jit(e_x: &Expr, e_y: &Expr, x: &mut NumT, y: &mut NumT) -> bool {
    let f_x = e_x.jit.as_ref().map(|j| j.func()).expect("No JIT for e_x");
    let f_y = e_y.jit.as_ref().map(|j| j.func()).expect("No JIT for e_y");
    unsafe {
        f_x(x, y);
        f_y(y, x);
    }
    true
}

/// Test a pair with specific initial values, returns true if matches ANSWER
fn test_pair_with_init(e_x: &Expr, e_y: &Expr, init_x: NumT, init_y: NumT) -> bool {
    let mut x = init_x;
    let mut y = init_y;
    let mut matcher = Matcher::new();
    
    for (i, _) in ANSWER.iter().enumerate() {
        let ok = if USE_JIT {
            eval_jit(e_x, e_y, &mut x, &mut y)
        } else {
            eval_naive(e_x, e_y, &mut x, &mut y)
        };
        
        if !ok || !matcher.match_one(i, x) {
            return false;
        }
    }
    matcher.match_final(e_x, e_y)
}

/// Test a pair against all initial value combinations, returns Some((init_x, init_y)) if found
fn test_pair(e_x: &Expr, e_y: &Expr) -> Option<(NumT, NumT)> {
    for init_x in INIT_X_MIN..=INIT_X_MAX {
        for init_y in INIT_Y_MIN..=INIT_Y_MAX {
            if test_pair_with_init(e_x, e_y, init_x, init_y) {
                return Some((init_x, init_y));
            }
        }
    }
    None
}

fn print_result(e_x: &Expr, e_y: &Expr, init_x: NumT, init_y: NumT) {
    use std::io::Write;
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    write!(lock, "x={}, y={} : ", init_x, init_y).unwrap();
    drop(lock);
    
    print_expression(e_x, &['x', 'y']);
    print!("; ");
    print_expression(e_y, &['y', 'x']);
    println!();
}

fn gen_expressions(ctx: &mut Context, n: usize) {
    let mut en = HashSet::new();

    // Length 1: variables and literals
    if n == 1 {
        let vars = [Operator::Var, Operator::VarY];
        for &op in &vars {
            let e = make_expr(None, None, -1, op);
            let arc = Arc::new(e);
            en.insert(arc.clone());
            ctx.var_expressions.push(arc);
        }
        
        for &lit in LITERALS {
            cache_expression(&mut en, make_expr(None, None, lit, Operator::Literal));
        }
    }

    // Length 3: increment/decrement operators
    if n == 3 {
        if let Some(exprs_1) = ctx.expressions.get(&1) {
            for e_r in exprs_1 {
                if e_r.op != Operator::Literal {
                    for incdec in INCDEC_OPERATORS {
                        cache_expression(&mut en, make_expr(None, Some(e_r.clone()), -1, incdec.op));
                    }
                }
            }
        }
    }

    // Generate binary expressions
    {
        let expressions_ref = &ctx.expressions;

        // Generate binary ops for a given length split
        let gen_binary_ops = |n_l: usize, op_len: usize| -> Vec<Expr> {
            let mut local_exprs = Vec::new();
            let n_r = n.saturating_sub(n_l + op_len);
            if n_r < 1 { return local_exprs; }
            
            if let (Some(exprs_l), Some(exprs_r)) = (expressions_ref.get(&n_l), expressions_ref.get(&n_r)) {
                for e_l in exprs_l {
                    for e_r in exprs_r {
                        if PRUNE_CONST_EXPR && e_l.op == Operator::Literal && e_r.op == Operator::Literal {
                            continue;
                        }

                        let op_l_val = e_l.op as u8;
                        let op_r_val = e_r.op as u8;

                        // Check each enabled binary operator
                        for bin_op in binary_ops_by_len(op_len) {
                            // Apply precedence rules based on operator type
                            let can_apply = match bin_op.op {
                                Operator::BitOr => op_l_val >= 0x40 && op_r_val >= 0x50,
                                Operator::BitXor => op_l_val >= 0x50 && op_r_val >= 0x60,
                                Operator::BitAnd => op_l_val >= 0x60 && op_r_val >= 0x70,
                                Operator::Eq | Operator::Neq => op_l_val >= 0x70 && op_r_val >= 0x80,
                                Operator::Lt | Operator::Gt => op_l_val >= 0x80 && op_r_val >= 0x90,
                                Operator::Leq | Operator::Geq => op_l_val >= 0x80 && op_r_val >= 0x90,
                                Operator::BitShl | Operator::BitShr => op_l_val >= 0x90 && op_r_val >= 0xA0,
                                Operator::Add | Operator::Sub => op_l_val >= 0xA0 && op_r_val >= 0xB0,
                                Operator::Mul | Operator::Div | Operator::Mod | Operator::Pow => {
                                    op_l_val >= 0xB0 && op_r_val >= 0xC0 && e_r.literal != 1
                                }
                                _ => false,
                            };
                            
                            if can_apply {
                                local_exprs.push(make_expr(Some(e_l.clone()), Some(e_r.clone()), -1, bin_op.op));
                            }
                        }
                    }
                }
            }
            local_exprs
        };

        // 1-byte operators
        if n > 2 {
            let new_exprs: Vec<Expr> = if USE_MULTITHREAD {
                (1..(n - 1)).into_par_iter().flat_map(|n_l| gen_binary_ops(n_l, 1)).collect()
            } else {
                (1..(n - 1)).flat_map(|n_l| gen_binary_ops(n_l, 1)).collect()
            };

            if USE_MULTITHREAD {
                en.par_extend(new_exprs.into_par_iter().map(Arc::new));
            } else {
                en.extend(new_exprs.into_iter().map(Arc::new));
            }
        }

        // 2-byte operators
        if n > 3 {
            let new_exprs: Vec<Expr> = if USE_MULTITHREAD {
                (1..(n - 2)).into_par_iter().flat_map(|n_l| gen_binary_ops(n_l, 2)).collect()
            } else {
                (1..(n - 2)).flat_map(|n_l| gen_binary_ops(n_l, 2)).collect()
            };

            if USE_MULTITHREAD {
                en.par_extend(new_exprs.into_par_iter().map(Arc::new));
            } else {
                en.extend(new_exprs.into_iter().map(Arc::new));
            }
        }

        // Unary operators
        if n > 1 {
            if let Some(exprs_r) = expressions_ref.get(&(n - 1)) {
                for e_r in exprs_r {
                    if (e_r.op as u8) >= 0xC0 {
                        for unary_op in UNARY_OPERATORS {
                            cache_expression(&mut en, make_expr(None, Some(e_r.clone()), -1, unary_op.op));
                        }
                    }
                }
            }
        }

        // Parentheses
        if USE_PARENS && n > 2 {
            if let Some(exprs_r) = expressions_ref.get(&(n - 2)) {
                for e_r in exprs_r {
                    cache_expression(&mut en, make_expr(None, Some(e_r.clone()), -1, Operator::Parens));
                }
            }
        }
    }
    
    ctx.expressions.insert(n, en);
}

// =============================================================================
// STATEMENT GENERATION
// =============================================================================

fn gen_statements(ctx: &mut Context, n: usize) {
    if ctx.expressions.get(&n).is_none() {
        return;
    }
    
    let mut sn = HashSet::new();
    let e_l = if !ctx.var_expressions.is_empty() { 
        ctx.var_expressions[0].clone() 
    } else { 
        return; 
    };

    let expressions_ref = &ctx.expressions;

    // Generate statements for each assignment operator length
    for assign_op in ASSIGN_OPERATORS {
        let expr_len = n.saturating_sub(assign_op.len);
        if expr_len < 1 { continue; }
        
        if let Some(exprs_r) = expressions_ref.get(&expr_len) {
            if USE_MULTITHREAD && exprs_r.len() > 100 && assign_op.len == 1 {
                // Parallel for large sets with 1-byte operator
                let new_stmts: Vec<Expr> = exprs_r.par_iter().map(|e_r| {
                    make_expr(Some(e_l.clone()), Some(e_r.clone()), -1, assign_op.op)
                }).collect();
                sn.par_extend(new_stmts.into_par_iter().map(Arc::new));
            } else {
                for e_r in exprs_r {
                    cache_expression(&mut sn, make_expr(Some(e_l.clone()), Some(e_r.clone()), -1, assign_op.op));
                }
            }
        }
    }

    ctx.statements.insert(n, sn);
}

// =============================================================================
// SEARCH FUNCTIONS
// =============================================================================

fn dfs_search(ctx: &Context, target_n: usize) {
    let var_x = &ctx.var_expressions[0];
    
    // Generate statements of length target_n on-the-fly
    let gen_stmts_for_expr = |expr: &Arc<Expr>, expr_len: usize| -> Vec<Expr> {
        let mut stmts = Vec::new();
        
        for assign_op in ASSIGN_OPERATORS {
            if expr_len + assign_op.len == target_n {
                stmts.push(make_statement(var_x, expr, assign_op.op));
            }
        }
        
        stmts
    };
    
    // Collect all cached statements for y
    let cached_stmts_y: Vec<&Arc<Expr>> = (1..=MAX_CACHE_LENGTH)
        .filter_map(|len| ctx.statements.get(&len))
        .flat_map(|s| s.iter())
        .collect();
    
    if USE_MULTITHREAD {
        (1..=MAX_CACHE_LENGTH).into_par_iter().for_each(|expr_len| {
            if let Some(exprs) = ctx.expressions.get(&expr_len) {
                exprs.par_iter().for_each(|expr| {
                    let stmts_x = gen_stmts_for_expr(expr, expr_len);
                    for stmt_x in &stmts_x {
                        for stmt_y in &cached_stmts_y {
                            if let Some((ix, iy)) = test_pair(stmt_x, stmt_y) {
                                print_result(stmt_x, stmt_y, ix, iy);
                            }
                        }
                        
                        for y_expr_len in 1..=MAX_CACHE_LENGTH {
                            if let Some(y_exprs) = ctx.expressions.get(&y_expr_len) {
                                for y_expr in y_exprs {
                                    let stmts_y = gen_stmts_for_expr(y_expr, y_expr_len);
                                    for stmt_y in &stmts_y {
                                        if let Some((ix, iy)) = test_pair(stmt_x, stmt_y) {
                                            print_result(stmt_x, stmt_y, ix, iy);
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }
        });
    } else {
        for expr_len in 1..=MAX_CACHE_LENGTH {
            if let Some(exprs) = ctx.expressions.get(&expr_len) {
                for expr in exprs {
                    let stmts_x = gen_stmts_for_expr(expr, expr_len);
                    for stmt_x in &stmts_x {
                        for stmt_y in &cached_stmts_y {
                            if let Some((ix, iy)) = test_pair(stmt_x, stmt_y) {
                                print_result(stmt_x, stmt_y, ix, iy);
                            }
                        }
                        
                        for y_expr_len in 1..=MAX_CACHE_LENGTH {
                            if let Some(y_exprs) = ctx.expressions.get(&y_expr_len) {
                                for y_expr in y_exprs {
                                    let stmts_y = gen_stmts_for_expr(y_expr, y_expr_len);
                                    for stmt_y in &stmts_y {
                                        if let Some((ix, iy)) = test_pair(stmt_x, stmt_y) {
                                            print_result(stmt_x, stmt_y, ix, iy);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn search_cached(ctx: &Context, max_n: usize) {
    if USE_MULTITHREAD {
        (1..=max_n.min(MAX_CACHE_LENGTH)).into_par_iter().for_each(|n_x| {
            if let Some(stmts_x) = ctx.statements.get(&n_x) {
                stmts_x.par_iter().for_each(|e_x| {
                    for n_y in 1..=max_n.min(MAX_CACHE_LENGTH) {
                        if let Some(stmts_y) = ctx.statements.get(&n_y) {
                            for e_y in stmts_y {
                                if let Some((ix, iy)) = test_pair(e_x, e_y) {
                                    print_result(e_x, e_y, ix, iy);
                                }
                            }
                        }
                    }
                });
            }
        });
    } else {
        for n_x in 1..=max_n.min(MAX_CACHE_LENGTH) {
            if let Some(stmts_x) = ctx.statements.get(&n_x) {
                for e_x in stmts_x {
                    for n_y in 1..=max_n.min(MAX_CACHE_LENGTH) {
                        if let Some(stmts_y) = ctx.statements.get(&n_y) {
                            for e_y in stmts_y {
                                if let Some((ix, iy)) = test_pair(e_x, e_y) {
                                    print_result(e_x, e_y, ix, iy);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    println!("Expression Search");
    println!("=================");
    println!("Target: {:?}", ANSWER);
    println!("Init: x=[{}..={}], y=[{}..={}]", INIT_X_MIN, INIT_X_MAX, INIT_Y_MIN, INIT_Y_MAX);
    println!("Max length: {}, Cache length: {}", MAX_LENGTH, MAX_CACHE_LENGTH);
    println!("JIT: {}, Multithread: {}", USE_JIT, USE_MULTITHREAD);
    println!("Binary ops: {}, Unary ops: {}, Assign ops: {}", 
             BINARY_OPERATORS.len(), UNARY_OPERATORS.len(), ASSIGN_OPERATORS.len());
    println!();

    let start = Instant::now();
    let mut ctx = Context::new();

    // Phase 1: Generate and cache expressions up to MAX_CACHE_LENGTH
    for n in 1..=MAX_CACHE_LENGTH {
        println!("Finding length {}...", n);
        gen_expressions(&mut ctx, n);
        gen_statements(&mut ctx, n);
        search_cached(&ctx, n);

        let expr_count = ctx.expressions.get(&n).map(|s| s.len()).unwrap_or(0);
        let stmt_count = ctx.statements.get(&n).map(|s| s.len()).unwrap_or(0);
        println!("  {} expressions, {} statements", expr_count, stmt_count);
        println!("  time: {:.3}s", start.elapsed().as_secs_f64());
    }
    
    // Phase 2: DFS search for lengths beyond MAX_CACHE_LENGTH
    for n in (MAX_CACHE_LENGTH + 1)..=MAX_LENGTH {
        println!("Finding length {}-{} (DFS)...", n, MAX_LENGTH);
        dfs_search(&ctx, n);
        println!("  time: {:.3}s", start.elapsed().as_secs_f64());
    }
    
    println!("\nDone! Total time: {:.3}s", start.elapsed().as_secs_f64());
}
