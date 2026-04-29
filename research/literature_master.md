# Master Literature File (TFM)

Status: deep synthesis and project-aligned knowledge base  
Last rebuilt: 2026-04-24  
Scope: semantic and temporal scientific cartography, with an optional explainable prediction layer  
Goal: keep the literature in one dense file that is fast to reuse later while still going beyond a list of citations

---

# 1. What the literature says in one page

## 1.1 Main conclusion for the TFM

The literature strongly supports the idea that this project should **not** be framed as "just making a pretty paper map". The strongest version of the thesis is:

1. build a semantic representation of a coherent scientific corpus
2. turn that representation into a stable and readable cartographic structure
3. extract region-level and paper-level variables from the map
4. analyze how those variables evolve through time
5. test whether they add explanatory or predictive value beyond text and metadata

That framing is much better supported than a pure visualization project.

## 1.2 The most important cross-paper insights

1. There is no single "true" science map. Different maps capture different relations: semantic similarity, shared references, shared citers, topic mixtures, or citation flow.
2. A raw 2D projection is not enough. The literature repeatedly shows that readability comes from the whole pipeline: representation, graph construction, clustering, aggregation, labeling, and temporal consistency.
3. Region extraction should come from graph/community structure, not only from visual islands in UMAP or t-SNE.
4. Citation-based structure is useful but lagging. Semantic structure is earlier and more flexible. The best TFM should compare or combine both.
5. Temporal analysis is a major opportunity. Drift, growth, burstiness, and fragmentation are well-grounded research targets.
6. Novelty is not one thing. The literature distinguishes atypical combinations, new combinations, content novelty, context novelty, and disruption.
7. Prediction should use careful targets. Raw current citation counts are convenient but methodologically weak; fixed-horizon or normalized targets are better.
8. SHAP is a defensible choice if the predictive layer is tree-based, especially when explanations are aggregated by feature family.

## 1.3 What this means for your thesis design

If time is limited, the literature points to the following as the most defensible core:

1. semantic representation with a strong scientific-document encoder
2. graph-based regionalization of the field
3. temporal analysis of region evolution
4. optional predictive experiment where cartographic features are compared against text/metadata baselines

If the predictive block underperforms, the thesis still remains strong as a study of cartographic representation and temporal field dynamics.

---

# 2. Coverage map from `ideas_database.md` to literature

| Idea block from the repo | Literature anchors | What they justify |
| --- | --- | --- |
| Cartography as analytical representation | Borner 2003, Chen 2006, Waltman 2010 | A map is an analytical object, not only a visual output |
| Semantic representation | SciBERT, SPECTER, SciNCL, SPECTER2 | How to embed scientific papers beyond bag-of-words |
| Topic labeling and summaries | LDA, RTM, BERTopic, DTM, D-ETM | How to label regions and track semantic shifts |
| Readable atlas construction | Fruchterman-Reingold, ForceAtlas2, UMAP, Leiden, Infomap | How to turn paper clouds into readable map units |
| Temporal atlas and region change | Leydesdorff 2008, Cobo 2011, Small 2006, Upham 2010 | Drift, growth, emergence, fragmentation |
| Novelty and impact | Uzzi 2013, Wang 2013, Wu 2019, Park 2023, Shi and Evans 2023 | Novelty, disruption, long-run impact, frontier behavior |
| Heterogeneity and diversity | Hofstra 2020, Shi and Evans 2023 | Why local mixture and cross-field combination matter |
| Prediction of future impact | Li 2019, SChuBERT 2020, Holm 2020, Hirako 2023 | How citation prediction is usually framed and where it fails |
| Explainability | XGBoost 2016, LIME 2016, SHAP 2017, TreeSHAP 2020 | How to explain feature contributions cleanly |

Important note:
some ideas in `ideas_database.md` are design choices rather than literature-defined methods. This is especially true for:

1. hexagonal map rendering
2. continuous density surfaces
3. landscape/relief metaphors
4. zoom-dependent UI behaviors

The literature supports these indirectly through aggregation and readability arguments, but they are not themselves the scientific contribution. They should be presented as cartographic encodings layered on top of a stronger analytical pipeline.

---

# 3. Classical science mapping and bibliometric structure

## 3.1 [Kessler (1963) - Bibliographic coupling between scientific papers](https://doi.org/10.1002/asi.5090140103)

- Core idea: two papers are related if they cite the same earlier works.
- Methodological contribution: gives a backward-looking similarity signal available immediately when a paper appears.
- Main value: excellent for early structure, because it does not need future citations to accumulate.
- For this TFM: use bibliographic coupling as one graph layer or as a validation signal against semantic similarity.
- Limitation: reference practices vary by field and style; semantically similar papers can look distant if they cite different canons.

## 3.2 [Small (1973) - Co-citation in the scientific literature](https://doi.org/10.1002/asi.4630240406)

- Core idea: two papers are related if later papers cite them together.
- Methodological contribution: defines a forward-looking structural relation that often recovers specialties and intellectual neighborhoods.
- Main value: co-citation is often more stable and historically meaningful than purely textual similarity.
- For this TFM: useful for validating mature regions and detecting influential or canonical hubs.
- Limitation: it is slow to form, so it is weaker for newly published papers and frontier detection.

## 3.3 [Borner, Chen, and Boyack (2003) - Visualizing knowledge domains](https://doi.org/10.1002/aris.1440370106)

- Core idea: science mapping is a pipeline problem, not a single-algorithm problem.
- Methodological contribution: surveys data models, similarity measures, clustering, layout, interaction, and evaluation.
- Main value: legitimizes the thesis as a methodological study of representation choices.
- For this TFM: this is one of the best high-level anchors for the "why map science?" chapter.
- Limitation: it is broad and foundational rather than operational; it tells you what matters, not a final recipe.

## 3.4 [Chen (2006) - CiteSpace II](https://doi.org/10.1002/asi.20317)

- Core idea: emerging trends can be detected from time-sliced citation structure, burst detection, and pivotal nodes.
- Methodological contribution: combines co-citation clusters, timeline views, betweenness centrality, and burstiness.
- Main value: shifts the map from static geography to evolving frontiers.
- For this TFM: supports using burst-based and structural-transition features when analyzing regions over time.
- Limitation: CiteSpace is still citation-centered, so it may underrepresent semantic novelty that is not yet highly cited.

## 3.5 [Waltman, van Eck, and Noyons (2010) - A unified approach to mapping and clustering of bibliometric networks](https://doi.org/10.1016/j.joi.2010.07.002)

- Core idea: mapping and clustering should come from a consistent mathematical objective rather than unrelated steps.
- Methodological contribution: shows that VOS-style mapping and modularity-style clustering can be derived from a common principle.
- Main value: avoids contradictory outputs where map geometry says one thing and cluster labels say another.
- For this TFM: very strong justification for coupling spatialization and region extraction conceptually.
- Limitation: it is still network-centered, so the quality of the final map depends heavily on how the graph is built.

## 3.6 [van Eck and Waltman (2010) - VOSviewer](https://doi.org/10.1007/s11192-009-0146-3)

- Core idea: bibliometric maps need software that emphasizes readability and interactive exploration, not only computation.
- Methodological contribution: packages VOS mapping, clustering, and several visualization modes for bibliometric data.
- Main value: shows what a usable bibliometric atlas looks like in practice.
- For this TFM: a strong baseline for rapid mapping and for checking whether custom methods really improve interpretability.
- Limitation: it is a great baseline but not the scientific novelty of your thesis.

## 3.7 [Cobo et al. (2011) - Science mapping software tools: Review, analysis, and cooperative study among tools](https://doi.org/10.1002/asi.21525)

- Core idea: different science-mapping tools emphasize different assumptions, workflows, and outputs.
- Methodological contribution: compares tools in terms of preprocessing, network construction, clustering, visualization, and temporal support.
- Main value: supports the thesis argument that tool choice is a methodological decision, not a neutral implementation detail.
- For this TFM: useful when explaining why you prefer custom pipelines plus selective external baselines.
- Limitation: the paper is comparative rather than algorithmically deep.

### Synthesis for this section

The classical bibliometric literature says that your atlas should combine:

1. a strong relational signal
2. a principled mapping/clustering pipeline
3. a temporal lens
4. explicit evaluation of interpretability and stability

That is exactly the direction of `ideas_database.md` sections 1, 2, and 9.

---

# 4. Semantic document representation and topic extraction

## 4.1 [Beltagy, Lo, and Cohan (2019) - SciBERT: A Pretrained Language Model for Scientific Text](https://aclanthology.org/D19-1371/)

- Core idea: scientific text differs enough from generic text that in-domain language model pretraining matters.
- Methodological contribution: trains BERT on a large scientific corpus and evaluates it on multiple downstream tasks.
- Main value: establishes a strong scientific-text baseline that is better aligned to abstracts and titles than generic BERT.
- For this TFM: good baseline encoder if you want a simpler text-only representation before moving to citation-informed models.
- Limitation: SciBERT is not optimized specifically for document-level similarity between papers.

## 4.2 [Reimers and Gurevych (2019) - Sentence-BERT](https://aclanthology.org/D19-1410/)

- Core idea: a model trained for efficient semantic similarity is better suited to retrieval and clustering than raw BERT embeddings.
- Methodological contribution: uses siamese/triplet objectives to make cosine similarity meaningful.
- Main value: a strong generic baseline for semantic retrieval and clustering.
- For this TFM: useful as a non-scientific-domain baseline to test how much domain specialization really helps.
- Limitation: it is not trained on scientific papers or citation relations, so it may miss scholarly similarity cues.

## 4.3 [Cohan et al. (2020) - SPECTER](https://aclanthology.org/2020.acl-main.207/)

- Core idea: scientific-paper embeddings improve when training uses citation structure, not only text.
- Methodological contribution: pretrains a Transformer with citation-informed triplets and evaluates on scientific-document tasks.
- Main value: one of the clearest papers showing that scientific similarity is not purely lexical.
- For this TFM: one of the strongest candidates for the primary semantic representation of papers.
- Limitation: direct citation supervision can encode citation biases and may overfit to citation-derived notions of relatedness.

## 4.4 [Ostendorff et al. (2022) - Neighborhood Contrastive Learning for Scientific Document Representations with Citation Embeddings](https://aclanthology.org/2022.emnlp-main.802/)

- Core idea: direct citations are too sparse and too binary to define scientific similarity well.
- Methodological contribution: samples positives and negatives from a citation-embedding neighborhood rather than only from direct citation edges.
- Main value: reframes similarity as continuous neighborhood structure instead of exact citation links.
- For this TFM: highly relevant if you want paper embeddings that support local-density and neighborhood-based cartographic variables.
- Limitation: still depends on citation graph quality and on the assumptions built into the graph embedding.

## 4.5 [Singh et al. (2023) - SciRepEval and SPECTER2](https://aclanthology.org/2023.emnlp-main.338/)

- Core idea: a single scientific-document embedding is often not equally good for classification, ranking, regression, and search.
- Methodological contribution: introduces a broader benchmark and shows that multi-format representations generalize better.
- Main value: a direct warning against assuming that one embedding quality metric is enough.
- For this TFM: if you choose one representation for both cartography and prediction, this paper is essential because it argues for broader evaluation.
- Limitation: benchmark breadth improves rigor but also makes model selection less simple.

## 4.6 [Blei, Ng, and Jordan (2003) - Latent Dirichlet Allocation](https://jmlr.org/papers/v3/blei03a.html)

- Core idea: each document can be represented as a mixture of latent topics and each topic as a distribution over words.
- Methodological contribution: provides the canonical probabilistic topic model for large text corpora.
- Main value: topics are interpretable and easy to summarize, which makes them useful for labels and thematic overviews.
- For this TFM: LDA is more useful for naming regions and summarizing corpora than for defining the final geometry of the atlas.
- Limitation: bag-of-words assumptions ignore word order and modern semantic similarity.

## 4.7 [Chang and Blei (2009) - Relational Topic Models for Document Networks](https://proceedings.mlr.press/v5/chang09a.html)

- Core idea: document content and document links should be modeled together.
- Methodological contribution: jointly models words and link structure so that topics help explain citations or other document relations.
- Main value: conceptually bridges semantic modeling and network structure.
- For this TFM: very useful as a theoretical precedent for hybrid semantic-plus-citation cartography.
- Limitation: topic-model assumptions are often less competitive than modern embedding methods for dense semantic geometry.

## 4.8 [Grootendorst (2022) - BERTopic](https://arxiv.org/abs/2203.05794)

- Core idea: modern topic modeling can be treated as embedding + clustering + class-based TF-IDF labeling.
- Methodological contribution: uses transformer embeddings and c-TF-IDF to produce interpretable topic summaries.
- Main value: topic quality often improves when semantic embeddings replace bag-of-words-only assumptions.
- For this TFM: BERTopic is a strong choice for region naming, local summaries, and exploratory thematic diagnostics.
- Limitation: the clusters depend strongly on the embedding and clustering pipeline; topics are not uniquely defined.

### Synthesis for this section

The semantic-representation literature suggests a practical hierarchy:

1. use SPECTER or SPECTER2 as the leading candidate for paper-level geometry
2. keep SciBERT or SBERT as baselines
3. use BERTopic or LDA for labels and summaries, not as the only source of map geometry
4. keep hybrid text-plus-link thinking in mind through RTM and SciNCL

This section directly supports `ideas_database.md` sections 2, 4, 8, and 9.

---

# 5. Layout, projection, region extraction, and readable map units

## 5.1 [Fruchterman and Reingold (1991) - Graph drawing by force-directed placement](https://doi.org/10.1002/spe.4380211102)

- Core idea: readable graph layouts arise from a balance of attraction and repulsion.
- Methodological contribution: one of the canonical force-directed layout formulations.
- Main value: establishes the logic behind spatializing relation graphs as readable maps.
- For this TFM: a historical baseline and a conceptual ancestor of later atlas-like layouts.
- Limitation: not ideal as the only layout method for large, dense scientific corpora.

## 5.2 [Jacomy et al. (2014) - ForceAtlas2](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679)

- Core idea: graph layouts should be continuous, interactive, and practical enough for exploratory analysis.
- Methodological contribution: improves usability, speed, and tuning over older force layouts.
- Main value: produces layouts that often look much more like an "atlas" than raw manifold projections.
- For this TFM: a strong candidate for graph-based map views, especially if you build a semantic or hybrid kNN graph.
- Limitation: attractive visual results do not automatically imply metric faithfulness.

## 5.3 [van der Maaten and Hinton (2008) - Visualizing Data using t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html)

- Core idea: preserve local neighborhoods when projecting high-dimensional data into 2D.
- Methodological contribution: defines a probabilistic neighbor-preservation objective with a heavy-tailed low-dimensional distribution.
- Main value: often creates visually separated clusters that help exploratory inspection.
- For this TFM: useful as an exploratory projection baseline.
- Limitation: distances between clusters and apparent region sizes are not trustworthy enough to define cartographic variables directly.

## 5.4 [McInnes, Healy, and Melville (2018) - UMAP](https://arxiv.org/abs/1802.03426)

- Core idea: preserve manifold neighborhood structure more efficiently and at larger scale than earlier methods.
- Methodological contribution: constructs fuzzy simplicial neighborhoods and optimizes a low-dimensional embedding.
- Main value: scalable, flexible, and strong for local structure.
- For this TFM: a practical default for projecting embeddings into 2D or 3D before adding map overlays.
- Limitation: hyperparameters strongly affect the visual map, so stability analysis is necessary.

## 5.5 [Blondel et al. (2008) - Louvain](https://doi.org/10.1088/1742-5468/2008/10/P10008)

- Core idea: large networks can be partitioned efficiently by greedy modularity optimization.
- Methodological contribution: scalable hierarchical community detection.
- Main value: became the default baseline for large graph regionalization.
- For this TFM: useful as a historical baseline when you compare region extraction methods.
- Limitation: can return poorly connected communities and is no longer the best default if robustness matters.

## 5.6 [Traag, Waltman, and van Eck (2019) - Leiden](https://doi.org/10.1038/s41598-019-41695-z)

- Core idea: community detection should guarantee well-connected communities, not only optimize modularity greedily.
- Methodological contribution: adds refinement and stronger guarantees over Louvain.
- Main value: one of the most defensible choices for region extraction in large document graphs.
- For this TFM: Leiden is the safest default for thematic regions and for temporal region tracking.
- Limitation: results still depend on graph construction and resolution choice.

## 5.7 [Rosvall and Bergstrom (2008) - Maps of random walks on complex networks reveal community structure](https://doi.org/10.1073/pnas.0706851105)

- Core idea: communities can be defined by information flow, not only edge density.
- Methodological contribution: introduces the map equation and flow-based community detection.
- Main value: gives an alternative notion of region structure that can differ meaningfully from modularity-based clusters.
- For this TFM: valuable as a robustness check if you want to see whether findings survive a different region definition.
- Limitation: can produce more fragmented or differently scaled partitions than modularity methods.

## 5.8 [van Eck et al. (2010) - A comparison of two techniques for bibliometric mapping: Multidimensional scaling and VOS](https://doi.org/10.1002/asi.21421)

- Core idea: not all projection methods distort bibliometric structure in the same way.
- Methodological contribution: compares classical MDS-style mapping with VOS optimization.
- Main value: warns against naive interpretation of pretty 2D scatterplots.
- For this TFM: strong support for explicitly evaluating projection artifacts before deriving map variables from coordinates.
- Limitation: focused on bibliometric maps, not modern transformer embedding spaces.

### Synthesis for this section

The literature points to a very clear practical rule:

1. use 2D projection for visualization
2. use graph communities for regions
3. use aggregation layers such as hex bins, density fields, or labeled regions for readability
4. never confuse visual separation with scientifically valid territorial structure unless stability tests support it

That directly supports `ideas_database.md` sections 3 and 4.

---

# 6. Temporal evolution, emergence, and region tracking

## 6.1 [Leydesdorff and Schank (2008) - Dynamic animations of journal maps](https://doi.org/10.1002/asi.20891)

- Core idea: time slices should be comparable, not independently optimized every year.
- Methodological contribution: balances within-slice fit and across-slice smoothness in dynamic mapping.
- Main value: gives a principled basis for animated or longitudinal maps.
- For this TFM: supports using anchored coordinates and measuring drift rather than letting each year reconfigure freely.
- Limitation: smoothness can hide real discontinuities if enforced too strongly.

## 6.2 [Chen, Ibekwe-SanJuan, and Hou (2010) - The structure and dynamics of cocitation clusters](https://doi.org/10.1002/asi.21309)

- Core idea: clusters become much more useful when they are interpreted from multiple views rather than only one algorithmic label.
- Methodological contribution: combines clustering, labeling, and summarization for dynamic co-citation analysis.
- Main value: region interpretation is part of the science, not an afterthought.
- For this TFM: useful for building a pipeline that names, summarizes, and tracks regions across time.
- Limitation: still rooted in co-citation data, which lags frontier activity.

## 6.3 [Cobo et al. (2011) - Detecting, quantifying, and visualizing the evolution of a research field](https://doi.org/10.1016/j.joi.2010.10.002)

- Core idea: themes can be described by structural indicators such as density and centrality, and then tracked over time.
- Methodological contribution: the strategic-diagram logic of motor themes, basic themes, and emerging or declining themes.
- Main value: gives a vocabulary for describing region maturity.
- For this TFM: very helpful when turning map regions into measurable thematic states.
- Limitation: co-word methods are interpretable but sensitive to term extraction quality.

## 6.4 [Small (2006) - Tracking and predicting growth areas in science](https://doi.org/10.1007/s11192-006-0132-y)

- Core idea: growth fronts can be detected before they dominate the field.
- Methodological contribution: uses citation dynamics to identify emerging areas.
- Main value: supports the thesis idea that frontiers can be operationalized quantitatively.
- For this TFM: motivates growth-rate, influx, and acceleration features at the region level.
- Limitation: success depends on the citation window and the granularity of the front definition.

## 6.5 [Upham and Small (2010) - Emerging research fronts in science and technology](https://doi.org/10.1007/s11192-009-0051-9)

- Core idea: not every expanding cluster is a genuine emerging front; emergence has identifiable structural patterns.
- Methodological contribution: studies how new research fronts form and consolidate.
- Main value: helps distinguish mere growth from meaningful thematic emergence.
- For this TFM: useful when defining stricter frontier criteria than just rising paper counts.
- Limitation: focused on citation-front logic rather than semantic neighborhoods.

## 6.6 [Blei and Lafferty (2006) - Dynamic Topic Models](https://www.cs.columbia.edu/~blei/papers/BleiLafferty2006a.pdf)

- Core idea: topics should be allowed to evolve through time rather than remain fixed.
- Methodological contribution: uses state-space dynamics on topic parameters to learn topic trajectories.
- Main value: foundational for diachronic thematic analysis.
- For this TFM: relevant if you want region labels or semantic summaries that evolve historically rather than staying static.
- Limitation: classic DTM is heavier and less semantically rich than embedding-based alternatives.

## 6.7 [Wang, Blei, and Heckerman (2008) - Continuous time dynamic topic models](https://proceedings.mlr.press/r6/wang08a.html)

- Core idea: topic change need not be modeled only in fixed discrete periods.
- Methodological contribution: extends dynamic topic models to continuous time.
- Main value: better suited to irregularly timed corpora.
- For this TFM: useful if you later move from year-buckets to publication-date-aware temporal modeling.
- Limitation: more complex and probably unnecessary for the first TFM version.

## 6.8 [Dieng, Ruiz, and Blei (2019) - The Dynamic Embedded Topic Model](https://arxiv.org/abs/1907.05545)

- Core idea: dynamic topic models benefit from modern embedding-based word and topic representations.
- Methodological contribution: combines dynamic topic trajectories with embedding-based topic parameterization.
- Main value: better topic coherence and diversity than older dynamic topic models on several corpora.
- For this TFM: probably the most relevant dynamic-topic paper if you want a modern temporal summarization layer.
- Limitation: still a topic model, so it is better for semantic evolution summaries than for final cartographic geometry.

### Synthesis for this section

This literature strongly supports the temporal part of your idea bank:

1. keep coordinates comparable through time
2. track region birth, drift, growth, shrinkage, and fragmentation
3. use dynamic topic summaries to explain what changed semantically
4. separate "emerging" from merely "recent" or "large"

This is the strongest non-predictive thesis block if the prediction layer later disappoints.

---

# 7. Novelty, disruption, heterogeneity, and long-term impact

## 7.1 [Uzzi et al. (2013) - Atypical combinations and scientific impact](https://doi.org/10.1126/science.1240474)

- Core idea: impactful science often mixes a conventional backbone with a selective atypical combination.
- Methodological contribution: measures novelty through unusual journal-pair combinations in references.
- Main value: one of the most influential operationalizations of scientific novelty.
- For this TFM: motivates combinatorial novelty features derived from references or neighborhoods.
- Limitation: journal-pair novelty is only one kind of novelty and can miss semantic surprise inside familiar venues.

## 7.2 [Wang, Song, and Barabasi (2013) - Quantifying long-term scientific impact](https://doi.org/10.1126/science.1237825)

- Core idea: citation trajectories can be decomposed into interpretable parameters such as fitness, immediacy, and longevity.
- Methodological contribution: models a paper's full citation history instead of reducing impact to one count.
- Main value: shows why long-term influence is different from short-term buzz.
- For this TFM: useful when defining targets or when interpreting why certain cartographic positions are associated with delayed impact.
- Limitation: the model is elegant but still reduces complex social processes to a small number of parameters.

## 7.3 [Hofstra et al. (2020) - The Diversity-Innovation Paradox in Science](https://doi.org/10.1073/pnas.1915378117)

- Core idea: marginalized groups and outsider positions often produce more novel work, but their contributions are discounted.
- Methodological contribution: links novelty, uptake, and career outcomes at scale.
- Main value: novelty and recognition are not the same thing.
- For this TFM: supports including institutional, demographic, or positional heterogeneity as explanatory context when possible.
- Limitation: this is about social inequality and innovation, not directly about map geometry.

## 7.4 [Ruan et al. (2021) - Rethinking the disruption index as a measure of scientific and technological advances](https://doi.org/10.1016/j.techfore.2021.121071)

- Core idea: disruption-style metrics are attractive but fragile.
- Methodological contribution: shows that disruption scores depend on reference counts and database coverage.
- Main value: a crucial cautionary paper for anyone tempted to use disruption as a clean indicator of novelty.
- For this TFM: if you use disruption, treat it as one noisy signal and report its methodological sensitivity.
- Limitation: it is mostly a critique, so it tells you more about what to avoid than what to build.

## 7.5 [Wu, Wang, and Evans (2019) - Large teams develop and small teams disrupt science and technology](https://doi.org/10.1038/s41586-019-0941-9)

- Core idea: team size systematically relates to whether work develops existing agendas or disrupts them.
- Methodological contribution: links team size, disruption behavior, search depth into prior literature, and attention timing.
- Main value: explains why team-related metadata may matter independently of textual content.
- For this TFM: team size should be included as a control variable if you model novelty or impact.
- Limitation: the result is aggregate and ecological; it does not mean every small team is disruptive.

## 7.6 [Park, Leahey, and Funk (2023) - Papers and patents are becoming less disruptive over time](https://doi.org/10.1038/s41586-022-05543-x)

- Core idea: over decades, science appears to rely on narrower portions of prior knowledge and becomes less disruptive on average.
- Methodological contribution: tracks disruption at massive scale across papers and patents.
- Main value: gives a macro-historical context for why frontier detection matters.
- For this TFM: justifies looking for shrinking frontier behavior or conservative local neighborhoods over time.
- Limitation: the interpretation depends on disruption metrics that remain debated.

## 7.7 [Shi and Evans (2023) - Surprising combinations of research contents and contexts are related to impact](https://doi.org/10.1038/s41467-023-36741-4)

- Core idea: novelty has at least two dimensions: content novelty and context novelty.
- Methodological contribution: models surprise through combinations of keywords and cited-journal contexts.
- Main value: one of the best modern papers for going beyond a single novelty number.
- For this TFM: directly supports feature families such as unusual semantic mixture, cross-region context jumps, and neighborhood heterogeneity.
- Limitation: the modeling setup is sophisticated and not trivial to reproduce fully in a TFM.

## 7.8 [Wang, Veugelers, and Stephan (2017) - Bias against novelty in science: A cautionary tale for users of bibliometric indicators](https://doi.org/10.1016/j.respol.2017.06.006)

- Core idea: novel work is often undervalued when evaluation uses short windows or simplistic bibliometric indicators.
- Methodological contribution: studies how novel combinations relate to delayed recognition.
- Main value: warns against treating early citation performance as the whole story.
- For this TFM: strongly supports fixed-horizon design choices and careful interpretation of low early citations in frontier regions.
- Limitation: it is partly an evaluation-policy paper, so it does not directly tell you how to build the map.

### Synthesis for this section

The novelty literature suggests that your cartographic variables should not stop at:

1. density
2. centroid distance
3. cluster label

You should also consider:

1. unusual local mixtures
2. bridge or cross-context behavior
3. team-size controls
4. delayed-recognition hypotheses
5. the difference between novelty, disruption, and long-term impact

This section is the theoretical basis for `ideas_database.md` sections 4.3, 4.4, 4.6, and 5.

---

# 8. Citation prediction and evaluation design

## 8.1 [Waltman (2016) - Citation Metrics: A Primer on How (Not) to Normalize](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002542)

- Core idea: citation comparisons are distorted by field, year, document type, and citation density.
- Methodological contribution: explains normalization principles and common mistakes.
- Main value: raw citation counts are not enough for evaluative claims.
- For this TFM: if you build prediction targets, prefer fixed windows and field/year-aware targets over unnormalized lifetime counts.
- Limitation: it is a methodological primer, not a predictive model paper.

## 8.2 [Li et al. (2019) - A Neural Citation Count Prediction Model based on Peer Review Text](https://aclanthology.org/D19-1497/)

- Core idea: information beyond the published paper text, such as peer reviews, can improve citation prediction.
- Methodological contribution: learns semantic features from review text and combines them with hand-crafted signals.
- Main value: shows that prediction quality depends strongly on which signals are available at decision time.
- For this TFM: a useful caution that any benchmark must define the prediction-time information boundary clearly.
- Limitation: peer review text is usually unavailable outside special datasets, so the method is not directly transferable to your corpus.

## 8.3 [van Dongen, Maillette de Buy Wenniger, and Schomaker (2020) - SChuBERT](https://aclanthology.org/2020.sdp-1.17/)

- Core idea: longer textual input and stronger scholarly document encoding can improve citation prediction.
- Methodological contribution: chunks scholarly documents and applies BERT-based encoding.
- Main value: demonstrates that richer document text can matter, not only metadata.
- For this TFM: supports comparing abstract-only versus richer textual representations if you later have access to full text.
- Limitation: full-text access is expensive and often incomplete; the practical TFM may remain abstract-centered.

## 8.4 [Holm et al. (2020) - Longitudinal Citation Prediction using Temporal Graph Neural Networks](https://arxiv.org/abs/2012.05742)

- Core idea: citation prediction is fundamentally temporal and relational, not just a static regression task.
- Methodological contribution: models citation trajectories over time using temporal graph structure and metadata.
- Main value: a strong modern reference for sequence-style impact prediction.
- For this TFM: conceptually validates using region growth, network position, and temporal neighborhood signals.
- Limitation: graph neural sequence models are much less interpretable than the tree-based models you likely want as the thesis centerpiece.

## 8.5 [Hirako, Sasano, and Takeda (2023) - Realistic Citation Count Prediction Task for Newly Published Papers](https://aclanthology.org/2023.findings-eacl.84/)

- Core idea: many citation prediction papers leak future information or assume signals unavailable at publication time.
- Methodological contribution: defines a realistic task that uses only information actually available when prediction is made.
- Main value: one of the best recent papers for setting up a fair benchmark.
- For this TFM: essential if you want your predictive layer to be methodologically defensible.
- Limitation: the stricter the setup, the harder the prediction task becomes.

### Synthesis for this section

The prediction literature implies four concrete rules:

1. define prediction time carefully
2. avoid future-information leakage
3. prefer normalized or fixed-horizon targets
4. compare simple interpretable baselines against richer models before escalating complexity

That aligns very well with your own benchmark notes already stored in `research/results/`.

---

# 9. Explainable modeling for the optional prediction layer

## 9.1 [Chen and Guestrin (2016) - XGBoost](https://doi.org/10.1145/2939672.2939785)

- Core idea: gradient-boosted trees are a powerful default for heterogeneous tabular data.
- Methodological contribution: scalable regularized boosting with sparsity-aware optimization and strong practical engineering.
- Main value: consistently strong performance on structured prediction tasks.
- For this TFM: probably the best main model for combining metadata, semantic, cartographic, and temporal feature families.
- Limitation: strong predictive power does not by itself make the model interpretable; explanation tooling is still needed.

## 9.2 [Ribeiro, Singh, and Guestrin (2016) - "Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)

- Core idea: complex models can be explained locally by fitting an interpretable surrogate around one prediction.
- Methodological contribution: introduces LIME and representative-example selection.
- Main value: helped make local post-hoc explanations mainstream.
- For this TFM: useful as a contrast method or sanity-check baseline for local explanation.
- Limitation: explanations can be unstable because they depend on perturbation design and local sampling choices.

## 9.3 [Lundberg and Lee (2017) - A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)

- Core idea: feature attributions should satisfy additive consistency properties grounded in Shapley values.
- Methodological contribution: defines SHAP as a unified additive explanation framework.
- Main value: gives a rigorous theoretical basis for local feature attribution.
- For this TFM: the most defensible general explanation framework if you want to claim which feature families matter.
- Limitation: generic SHAP can be computationally expensive without model-specific optimizations.

## 9.4 [Lundberg et al. (2020) - From local explanations to global understanding with explainable AI for trees](https://doi.org/10.1038/s42256-019-0138-9)

- Core idea: tree models can be explained both locally and globally using exact, efficient SHAP-style methods.
- Methodological contribution: TreeSHAP plus interaction values and global aggregation workflows.
- Main value: probably the single best explanation paper for a tree-based TFM model.
- For this TFM: lets you aggregate explanations by feature family, region, cohort, or impact band.
- Limitation: explanation quality still depends on the validity of the predictive setup and feature engineering.

### Synthesis for this section

If you keep the predictive block:

1. use a tree-based model such as XGBoost or a close equivalent
2. explain it with TreeSHAP
3. report feature-family contributions, not only individual variables
4. use LIME only as a secondary local baseline if you want a comparison

---

# 10. Practical recommendations for the TFM after reading this literature

## 10.1 Recommended core pipeline

The literature most strongly supports this pipeline:

1. corpus definition in a coherent macro-field
2. primary paper embedding with SPECTER or SPECTER2
3. baseline comparison with SciBERT or SBERT
4. graph construction from semantic kNN, optionally enriched with bibliographic coupling or citation structure
5. region extraction with Leiden
6. 2D visualization with UMAP or ForceAtlas2 depending on whether the main object is an embedding cloud or a graph
7. cartographic aggregation with regions, density, or hex bins for readability
8. temporal tracking with anchored coordinates and region IDs
9. optional predictive layer with normalized targets and tree-based models
10. SHAP aggregation by feature family

## 10.2 Recommended feature families from the literature

### Density and position

Use:

1. local semantic density
2. k-nearest neighbor count or distance profile
3. distance to region centroid
4. distance to region boundary
5. bridge score between regions

Justification:
these are the most natural cartographic abstractions of the graph/layout/community literature.

### Heterogeneity and mixture

Use:

1. local topic entropy
2. local institutional diversity
3. local country diversity
4. semantic spread of neighbors
5. cross-region neighborhood ratio

Justification:
these are supported by the novelty, context-surprise, and diversity literature.

### Novelty and frontier

Use:

1. unusual reference-combination score
2. semantic deviation from historical neighborhood
3. region newcomer influx
4. local growth acceleration
5. burst or sharp recent expansion measures

Justification:
these are the operational bridge between the Uzzi/Small/Shi/Wang line of work and your cartographic framework.

### Temporal region dynamics

Use:

1. centroid drift
2. region growth rate
3. fragmentation or consolidation
4. stability of membership
5. emergence score based on growth plus novelty plus cohesion

Justification:
these are the cleanest region-level outcomes supported by the temporal-mapping literature.

## 10.3 Recommended targets for the optional predictive block

Best choices:

1. log citations after a fixed future window
2. top 10 percent or top 20 percent impact within year and subfield
3. citation percentile within cohort

Avoid as the main target:

1. current lifetime raw citations without age normalization
2. exact long-horizon count without normalization
3. cross-field raw impact comparisons

Reason:
the prediction and normalization literature is clear that target definition can dominate apparent model quality.

## 10.4 What not to overclaim

1. Do not claim that UMAP islands are "continents" unless region stability survives parameter changes.
2. Do not claim that any one map is the objective structure of the field.
3. Do not claim that disruption is a clean measure of novelty without caveats.
4. Do not claim predictive success from benchmarks that leak future information.
5. Do not present visualization design choices as the scientific contribution.

## 10.5 If the prediction layer fails

The strongest non-predictive thesis, according to this literature, is:

1. construct a semantic and bibliometric atlas of a coherent scientific field
2. derive measurable cartographic variables from it
3. analyze how thematic regions evolve over time
4. characterize frontier zones through growth, drift, and novelty
5. show that cartographic representation reveals structure that plain scatterplots or topic lists miss

That is still a solid TFM.

---

# 11. Short reading order if you need to re-enter the topic quickly later

If you only have a few hours in the future, read in this order:

1. Borner, Chen, and Boyack (2003) for the big picture
2. Waltman, van Eck, and Noyons (2010) for mapping plus clustering logic
3. SPECTER (2020) and SciRepEval/SPECTER2 (2023) for scientific document representation
4. Traag et al. (2019) for Leiden and region extraction
5. Leydesdorff and Schank (2008) plus Cobo et al. (2011) for temporal evolution
6. Uzzi et al. (2013), Shi and Evans (2023), and Wang et al. (2017) for novelty
7. Wang, Song, and Barabasi (2013) and Hirako et al. (2023) for impact modeling and target design
8. Chen and Guestrin (2016) plus Lundberg et al. (2020) for the optional predictive layer

---

# 12. Bottom-line verdict on the project idea

After synthesizing this literature, the thesis idea is strongest when stated as:

**Do cartographic variables derived from the semantic and temporal environment of a paper help describe the structure and evolution of a scientific field, and do they add explanatory or predictive value beyond text and metadata?**

That question is well supported.

The weakest version of the project would be:

**Can I draw a cool map of papers?**

The literature does not support stopping there.

