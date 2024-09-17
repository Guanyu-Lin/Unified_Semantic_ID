Traceback (most recent call last):
  File "train_semantic_full.py", line 163, in <module>
    main()
  File "train_semantic_full.py", line 124, in main
    model = S3RecModel(args=args)
  File "/projects/eng/shared/jiaxuan/guanyu/srq/models.py", line 21, in __init__
    self.code_embeddings = nn.ModuleList([nn.Embedding(self.args.attribute_size, self.args.semantic_dim_size).from_pretrained(self.rq_model[str(c) + "._codebook.embed"][0]) for _ in range(self.args.codebook_size)])
  File "/projects/eng/shared/jiaxuan/guanyu/srq/models.py", line 21, in <listcomp>
    self.code_embeddings = nn.ModuleList([nn.Embedding(self.args.attribute_size, self.args.semantic_dim_size).from_pretrained(self.rq_model[str(c) + "._codebook.embed"][0]) for _ in range(self.args.codebook_size)])
NameError: name 'c' is not defined
Traceback (most recent call last):
  File "train_semantic_full.py", line 163, in <module>
    main()
  File "train_semantic_full.py", line 124, in main
    model = S3RecModel(args=args)
  File "/projects/eng/shared/jiaxuan/guanyu/srq/models.py", line 21, in __init__
    self.code_embeddings = nn.ModuleList([nn.Embedding(self.args.attribute_size, self.args.semantic_dim_size).from_pretrained(self.rq_model[str(c) + "._codebook.embed"][0]) for _ in range(self.args.codebook_size)])
  File "/projects/eng/shared/jiaxuan/guanyu/srq/models.py", line 21, in <listcomp>
    self.code_embeddings = nn.ModuleList([nn.Embedding(self.args.attribute_size, self.args.semantic_dim_size).from_pretrained(self.rq_model[str(c) + "._codebook.embed"][0]) for _ in range(self.args.codebook_size)])
NameError: name 'c' is not defined
