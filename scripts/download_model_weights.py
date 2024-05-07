def modelscope():
    from modelscope.utils.constant import Tasks
    from modelscope.pipelines import pipeline
    text_generation_zh = pipeline(task=Tasks.text_generation,
                                  model='baichuan-inc/Baichuan2-7B-Base',
                                  device_map='auto', model_revision='v1.0.2')


if __name__ == "__main__":
    modelscope()
