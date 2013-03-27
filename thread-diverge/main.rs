extern mod std;
use std::time::precise_time_s;

extern mod OpenCL;

use OpenCL::hl::*;
use OpenCL::vector::*;

const N: uint = 1024;

fn main() {
    let ctx = create_compute_context_types([GPU]);

    io::println(fmt!("Using device %s", ctx.device_name()));

    let r = rand::Rng();

    let A = vec::from_fn(N * N, |_| r.gen_f64());
    let A = Vector::from_vec(ctx, A);

    let B = vec::from_fn(N * N, |_| r.gen_f64());
    let B = Vector::from_vec(ctx, B);

    let C = vec::from_elem(N * N, 0f64);
    let C = Vector::from_vec(ctx, C);

    let result = io::read_whole_file_str(&path::Path("kernels.cl"));
    let source = match move result {
        Ok(move src) => src,
        Err(move e) => fail fmt!("Could not read file: %?", e)
    };

    let program = ctx.create_program_from_source(source);
    match program.build(ctx.device) {
        Ok(()) => (),
        Err(build_log) => {
            io::println("Error building OpenCL program!");
            io::println("");
            io::println(build_log);
            fail
        }
    }

    do_kernel(ctx, &program, "MyAdd", N, 256, &A, &B, &C);
    do_kernel(ctx, &program, "MyAdd_2D", (N, N), (16, 16), &A, &B, &C);
    do_kernel(ctx, &program, "MyAdd_2D_unweave", (N, N), (16, 16), &A, &B, &C);

    do_kernel(ctx, &program, "MyAdd_col", N, 256, &A, &B, &C);
    do_kernel(ctx, &program, "MyAdd_2D_col", (N, N), (16, 16), &A, &B, &C);
    do_kernel(ctx, &program, "MyAdd_2D_unweave_col", (N, N), (16, 16),
              &A, &B, &C);

    do_kernel(ctx, &program, "MyAdd_2D_nobranch", (N, N), (16, 16),
              &A, &B, &C);
    do_kernel(ctx, &program, "MyAdd_2D_col_nobranch", (N, N), (16, 16),
              &A, &B, &C);
}

fn do_kernel<I: KernelIndex>(
    ctx: @ComputeContext,
    program: &Program,
    name: &str,
    global: I,
    local: I,
    A: &Vector<f64>,
    B: &Vector<f64>,
    C: &Vector<f64>)
{
    let k = program.create_kernel(name);
    let time = time_kernel(ctx, &k, global, local, A, B, C);
    
    io::println(fmt!("%24s:\t%f msec/kernel", name, time));
}

fn time_kernel<I: KernelIndex>(
    ctx: @ComputeContext, k: &Kernel,
    global: I,
    local: I,
    A: &Vector<f64>,
    B: &Vector<f64>,
    C: &Vector<f64>) -> float
{
    const REP: uint = 500;

    k.set_arg(0, A);
    k.set_arg(1, B);
    k.set_arg(2, C);
    k.set_arg(3, &N);

    // Call the timer once to warm up.
    precise_time_s();

    // Do it once to avoid timing inconsistencies.
    ctx.enqueue_async_kernel(k, global, local).wait();

    // Loop one
    let start1 = precise_time_s();
    for REP.times || {
        ctx.enqueue_async_kernel(k, global, local).wait();
    }
    let stop1 = precise_time_s();

    let elapsed1 = stop1 - start1;

    // Loop two
    let start2 = precise_time_s();
    for REP.times || {
        ctx.enqueue_async_kernel(k, global, local).wait();
        ctx.enqueue_async_kernel(k, global, local).wait();
    }
    let stop2 = precise_time_s();

    let elapsed2 = stop2 - start2;

    (elapsed2 - elapsed1) / (REP as float) * 1000f
}
