import asyncio
from mcpuniverse.tracer.collectors import MemoryCollector
from mcpuniverse.benchmark.runner import BenchmarkRunner

async def test():
    print("🧪 Running benchmark with OpenRouter...")
    trace_collector = MemoryCollector()
    
    # Use relative path - BenchmarkRunner looks in mcpuniverse/benchmark/configs/
    benchmark = BenchmarkRunner("dummy/benchmark_1.yaml")
    
    # Run the benchmark
    results = await benchmark.run(trace_collector=trace_collector)
    
    print(f"\n✅ Benchmark complete!")
    print(f"Results: {results}")
    
    # Get traces
    for result in results:
        print(f"\nTask trace IDs: {result.task_trace_ids}")

asyncio.run(test())