from swift.llm import export_main, ExportArguments

args = ExportArguments(
    model='/data/shuang/models/Qwen2.5-7B-Instruct',
    adapters='/data/shuang/short-horizon/output/v9-20260118-225059/checkpoint-66',
    output_dir='/data/shuang/models/Qwen2.5-7B-Instruct-ToolGym-v9',
    merge_lora=True,
)

export_main(args)
