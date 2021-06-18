import conllu

import datasets

class UniversaldependenciesConfig(datasets.BuilderConfig):
    """BuilderConfig for Universal dependencies"""

    def __init__(self, **kwargs):
        super(UniversaldependenciesConfig, self).__init__(version=datasets.Version("2.7.0", ""), **kwargs)


class UniversalDependencies(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("2.7.0")
    BUILDER_CONFIGS = [UniversaldependenciesConfig()]
    BUILDER_CONFIG_CLASS = UniversaldependenciesConfig


    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "idx": datasets.Value("string"),
                    "tokens": datasets.Value("string"),
                    "tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "B",
                                "I",
                                "O",
                            ]
                        )
                    ),
                }
            ),
        )


    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # urls_to_download = {}
        # for split, address in _UD_DATASETS[self.config.name].items():
        #     urls_to_download[split] = []
        #     if isinstance(address, list):
        #         for add in address:
        #             urls_to_download[split].append(_PREFIX + add)
        #     else:
        #         urls_to_download[split].append(_PREFIX + address)

        # downloaded_files = dl_manager.download_and_extract(urls_to_download)
        splits = []

 
        splits.append(
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": 'test.conll'})
        )


        return splits


    def _generate_examples(self, filepath):
        id = 0
        # print(filepath)
        # for path in filepath:
        #     print(path)
        with open(filepath, "r", encoding="utf-8") as data_file:
            tokenlist = list(conllu.parse_incr(data_file,fields = ['id','form','tag']))
            for i,sent in enumerate(tokenlist):

                if i == 0:
                  print([token["tag"] for token in sent])
                if "sent_id" in sent.metadata:
                    idx = sent.metadata["sent_id"]
                else:
                    idx = id

                tokens = [token["form"] for token in sent]

                if "text" in sent.metadata:
                    txt = sent.metadata["text"]
                else:
                    txt = " ".join(tokens)

                yield id, {
                    "idx": str(idx),
                    "tokens": txt,
                    "tags": [token["tag"] for token in sent],
                }
                id += 1        
