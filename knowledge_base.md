# TFM Knowledge Base
## Semantic and temporal cartography of papers with an optional explainable predictive layer

> Status: context document for the repo and for agents
> Goal: condense the strategic, methodological, and technical exploration completed so far
> Last updated: 2026-04-23

---

# 1. General project idea

The original intuition was to build a **paper map** where spatial proximity represents semantic similarity, with a cartographic aesthetic inspired by projects such as Paperscape and NIH Maps. From there, several possible directions emerged:

1. **Primarily visual project**
   1. build a useful and navigable map
   2. show thematic regions, densities, borders, and temporal evolution
   3. build a public web demo

2. **Primarily analytical project**
   1. build a cartographic representation of scientific space
   2. extract cartographic variables for each paper
   3. study whether those variables help explain or predict future impact

3. **Hybrid project**
   1. the TFM is a quantitative and methodological study
   2. the web is a complementary prototype, not the core

Strategic conclusion:
**the TFM should be primarily a study**, not a website.
If the web is built, it should function as a **demonstrator**.

---

# 2. Initial diagnosis of the idea

Several problems were identified with the initial idea of "making a paper map":

1. If embeddings are projected to 2D with UMAP or t-SNE and then drawn as points, the result is usually a **dense and unreadable cloud**.
2. A plain 2D projection rarely generates clear "continents" or geography on its own.
3. Novelty cannot simply consist of "putting papers on a map".
4. A thesis based only on visualization risks being visually impressive but academically weak.
5. The predictive component may not have enough signal, so it should not be the only pillar of the project.

Important takeaway:
**the problem is not only how to represent embeddings, but how to transform an abstract space into a readable cartographic unit**.

---

# 3. What was learned from existing projects

## 3.1 Paperscape

Paperscape does not work like a normal embedding scatterplot.
Its strength is not a miraculous projection, but a design pipeline:

1. layout through an **n-body** style physical simulation
2. separation of papers through artificial forces
3. rendering the map through pre-generated **tiles**
4. multiple visual layers
5. multi-scale labels
6. atlas-like navigation

Key ideas worth reusing without copying:
1. think in terms of **multi-scale representation**
2. separate **layout** from **rendering**
3. use **density, regions, or tessellation**, not only points
4. introduce **thematic and temporal layers**
5. accept some semi-automatic or manual labeling component

## 3.2 NIH Maps

NIH Maps is compelling because it does not operate on a tiny niche or on the whole scientific universe, but on a broad yet coherent corpus. There:

1. NIH grants were used
2. topics were extracted from title and abstract
3. LDA and a graph layout were applied
4. a readable thematic geography was generated

Main lesson:
**to see continents and thematic cores, an intermediate scale is needed**
Neither a very small niche nor all of science is ideal.

## 3.3 Open Knowledge Maps, Litmaps, ResearchRabbit, Inciteful, MapAlex

Several literature visualization references were reviewed.
Conclusions:

1. many tools are really **networks or graphs**, not continuous maps
2. Paperscape remains unusual in its strong territorial feeling
3. Open Knowledge Maps comes closer to a **thematic map**
4. Litmaps and ResearchRabbit are useful as exploration references, but less as territorial cartography
5. the project should borrow the multi-scale atlas principle, not copy a specific interface

---

# 4. Decision on the TFM focus

## 4.1 Rejected option

Building a TFM whose essence is "making a website like Paperscape".

Problems:
1. risk of spending too much time on frontend work
2. difficulty in justifying the scientific contribution
3. possibility of ending up with a beautiful demo and weak results
4. strong dependence on the website working perfectly

## 4.2 Preferred option

Build a TFM whose core is:

**constructing and evaluating a useful scientific cartography, studying the properties of its regions, and optionally using those properties as predictive features**

The web would be an extra.

## 4.3 Strong TFM narrative

The consolidated narrative is:

1. papers are represented semantically
2. that representation is turned into a multi-scale map
3. the map allows local properties of the scientific environment to be defined
4. those properties are studied through time
5. it is evaluated whether they help explain or predict future impact
6. SHAP or another explanatory tool closes the interpretive loop

---

# 5. Project options that were explored

## 5.1 Purely visual option
### Paper atlas

Goal:
create a navigable map of a discipline or macro-field with thematic regions, zoom, time, and density layers.

Pros:
1. visually powerful
2. useful as a portfolio piece
3. attractive as a web demo

Cons:
1. weaker academic contribution unless paired with serious analysis
2. high risk of turning into an interface project

## 5.2 Purely analytical option
### Cartography as a feature generator

Goal:
use the map to extract variables such as local density, novelty, position relative to cores, local impact gradient, and so on.

Pros:
1. fits a TFM better
2. can be validated with models and metrics
3. enables interpretability

Cons:
1. less spectacular visually if the web side is reduced too much

## 5.3 Hybrid option, the chosen one
### Semantic and temporal cartography plus possible explainable prediction

Goal:
produce a methodological study of representation and cartographic analysis, and then test whether map-derived variables add predictive value.

Pros:
1. balance between science and visualization
2. the web has a clear role as a demonstrator
3. SHAP can justify whether cartography adds anything real

---

# 6. Literature and research lines reviewed

## 6.1 Classical science mapping
The literature on science maps was reviewed, and the conclusion was that there is no single true map.
Different methods capture different facets:
1. citation-based maps
2. content-based maps
3. co-citation or bibliographic coupling maps
4. topic modeling

Conclusion:
**if different maps do not match, that does not necessarily mean one is wrong**

## 6.2 Content-based cartography and embeddings
Works were reviewed in which scientific maps are built using semantic similarity from titles, abstracts, or embeddings.
Lesson:
1. yes, it is viable
2. it is not enough to project embeddings and draw them
3. clustering, layout, aggregation, and evaluation are needed

## 6.3 Temporal dynamics
Works on topic trajectories and diachronic embeddings were reviewed.
Lessons:
1. time should not be treated only as a slider
2. one can study thematic drift, growth, fragmentation, and centroid shifts
3. this aligns very well with the cartographic intuition of the project

## 6.4 Impact prediction
Works were reviewed on:
1. citation prediction
2. relative impact prediction
3. domain-specific models
4. using text and embeddings to anticipate scientific success

Conclusions:
1. the topic has academic value
2. raw citation counts should not be used without normalization
3. fixed windows and relative metrics by cohort are preferable

## 6.5 Explainability and SHAP
Works were reviewed where SHAP is used to interpret citation prediction models.
Central conclusion:
**SHAP is fully defensible in this domain**
The interesting twist would be applying it to **families of cartographic features**, not only traditional metadata.

---

# 7. Consolidated research question

Suggested main question:

**Do cartographic variables derived from the semantic and temporal environment of a paper provide additional information, beyond text and metadata, to analyze the structure of a scientific field and eventually predict its future impact?**

Alternative versions:

1. Can a multi-scale cartographic representation of scientific space capture interpretable and stable thematic structure?
2. Which strategies transform a cloud of papers into a readable and useful map?
3. Does the cartographic environment of a paper contain predictive signal about its future impact?

---

# 8. Working hypotheses

## Visual and methodological hypotheses
1. A multi-scale cartographic representation is more useful than a simple 2D scatterplot.
2. The thematic structure of the corpus will be more interpretable if aggregations such as regions, density, or hexagons are used.

## Temporal hypotheses
1. Thematic regions show observable growth, drift, and fragmentation.
2. The temporal layer will reveal structural change that is not visible in a static snapshot.

## Optional predictive hypotheses
1. Cartographic features will add something over a baseline using only text and metadata.
2. Temporal environmental features may contribute more than purely static spatial position.
3. SHAP will make it possible to identify which cartographic families matter most.

---

# 9. Proposed TFM structure

## 1. Introduction
1.1. Scientific information overload  
1.2. Motivation for scientific cartography  
1.3. Limitations of standard visualizations  
1.4. Objectives of the work  
1.5. Research question and hypotheses

## 2. State of the art
2.1. Classical science mapping  
2.2. Topic modeling and embeddings  
2.3. Semantic cartography  
2.4. Temporal dynamics of scientific fields  
2.5. Impact prediction  
2.6. Explainable AI in scientometrics

## 3. Data and corpus construction
3.1. Data source  
3.2. Definition of the field or subfield  
3.3. Corpus filters  
3.4. Temporal split  
3.5. Cleaning and deduplication

## 4. Semantic representation
4.1. Available text  
4.2. Embedding construction  
4.3. Comparison of representations  
4.4. Embedding model selection

## 5. Construction of cartographic space
5.1. Similarity graph  
5.2. Dimensionality reduction for visualization  
5.3. Thematic grouping  
5.4. Stability and readability evaluation  
5.5. Multi-scale strategies

## 6. Temporal analysis of the map
6.1. Maps by temporal window  
6.2. Region tracking  
6.3. Drift and fragmentation  
6.4. Local impact relief

## 7. Cartographic feature engineering
7.1. Density and position  
7.2. Impact relief  
7.3. Novelty and mixing  
7.4. Temporal variables  
7.5. Hybrid network variables

## 8. Predictive modeling
8.1. Target definition  
8.2. Baseline  
8.3. Model with cartography  
8.4. Model comparison  
8.5. Evaluation by subgroups

## 9. Explainability
9.1. Global SHAP  
9.2. SHAP by feature family  
9.3. Local SHAP on specific papers  
9.4. Interpretive discussion

## 10. Discussion and limitations
10.1. What the map really captures  
10.2. Where it fails  
10.3. Coverage biases  
10.4. Scalability and replicability

## 11. Web prototype
11.1. Purpose of the demonstrator  
11.2. Multi-scale design  
11.3. Filters, time, and exploration  
11.4. Relationship to the thesis core

---

# 10. The 2D representation problem

A central problem was identified:
simple projection of embeddings into 2D produces dense and unreadable clouds.

## 10.1 What should not be done
1. rely entirely on UMAP for map structure
2. draw all papers as equal points
3. expect clear continents to emerge automatically

## 10.2 What should be done
1. separate layout from visualization
2. use multi-scale representation
3. use regions, density, hexagons, or tessellation
4. name regions using semi-automatic labels
5. treat 2D only as a visual layer, not as the sole analytical basis

## 10.3 Possible visualization strategies
1. hexagonal map
2. continuous density map
3. region-based map
4. atlas by temporal layers
5. hybrid of regions plus points on zoom

---

# 11. Ideas for cartographic features by paper

## 11.1 Local density and position
1. local semantic density
2. distance to cluster centroid
3. distance to thematic boundary
4. isolation or peripherality
5. thematic entropy of the neighborhood

## 11.2 Impact relief
1. local average citations
2. deviation of the paper from its neighbors
3. local slope of the relief
4. local ruggedness
5. relative height within local cohort

## 11.3 Novelty and mixing
1. semantic rarity
2. mixture of subtopics among neighbors
3. distance to typical combinations
4. local heterogeneity
5. proximity to intermediate zones between regions

## 11.4 Temporal variables
1. recent growth of the neighborhood
2. displacement of the local centroid
3. stability or fragmentation
4. changing density
5. emergence of new sub-islands or clusters

## 11.5 Hybrid variables
1. centrality in the similarity graph
2. exposure to adjacent fields
3. size and institutional diversity of the environment
4. coherence between assigned topic and geometric environment

---

# 12. Prediction: its role inside the project

## 12.1 What was discussed
It was repeatedly considered whether prediction should be the center of the work.

Conclusion:
**no**
Prediction should be:
1. an additional validation layer
2. a way to test whether cartography adds something
3. a later block, not the first objective

## 12.2 Candidate targets
### Less recommended
1. total raw citation count
2. exact time to 100 citations

Problems:
1. too noisy
2. strong field effects
3. full temporal citation trajectories are not always available

### More recommended
1. log citations at 3 years
2. top 10 percent by citations at 3 years within cohort and subfield
3. cohort-relative impact
4. binary or ordinal high-impact classification

## 12.3 Suggested models
1. simple baselines
   1. logistic regression
   2. XGBoost
   3. CatBoost
2. main suggested model for interpretability
   1. CatBoost or XGBoost with SHAP
3. optional benchmark
   1. tabular neural network
   2. MLP on embeddings and features

## 12.4 Role of SHAP
SHAP is intended to:
1. measure the weight of feature families
2. show whether cartographic features add real value
3. close the argument of the TFM through interpretability

---

# 13. Decisions about domain or discipline

## 13.1 Options that were considered
1. computer vision
2. NLP
3. machine learning
4. neural networks
5. all AI
6. all science
7. statistics and probability

## 13.2 General lessons
1. a very small niche may be too narrow to show geography
2. all of science is too broad and heterogeneous
3. the best scale is usually a **coherent macro-area**

## 13.3 On computer vision
Pros:
1. visible thematic structure
2. rich temporal narrative
3. good visual material

Cons:
1. possibly too narrow if the goal is an NIH Maps style effect

## 13.4 On neural networks
Final assessment was positive.
Pros:
1. sufficient size
2. real internal breadth
3. strong temporal narrative
4. reasonable semantic coherence

Cons:
1. much vocabulary is shared across subareas
2. risk of a map less fragmented than expected

## 13.5 On statistics and probability
This was eventually proposed as the working corpus.
Likely reasons:
1. coherent subfield
2. manageable size
3. less chaotic than general AI
4. stronger methodological control

Risk:
it may be less visually spectacular than an AI macro-field, though perhaps more academically solid.

---

# 14. Discussion of data sources

## 14.1 arXiv

### What it provides
1. technical preprints
2. clear repository categories
3. often accessible PDFs
4. a very convenient corpus in STEM

### Limitations
1. it does not cover all science
2. it does not necessarily cover the whole field
3. it does not provide institutions or countries well
4. it does not provide temporal citation trajectories

### Main fields in the API
1. id
2. title
3. summary
4. authors
5. published
6. updated
7. primary_category
8. categories
9. links
10. optional doi
11. optional journal_ref
12. optional comment

Conclusion:
arXiv is good for text and preprints, but not as the only base if rich metadata is needed.

## 14.2 OpenAlex

### What it provides
1. works, authors, institutions, topics, sources, funders
2. citations and institutional metadata
3. hierarchical topics
4. powerful filters
5. much broader coverage
6. free snapshot
7. modern API very useful for building datasets

### Abstract
OpenAlex does have abstracts when available, but in the form of `abstract_inverted_index`, not always as clean plain text.

### Embeddings
OpenAlex has semantic search and a `has_embeddings` filter field, but it was not assumed to expose all per-paper embeddings conveniently as a primary pipeline input.

### Limitations
1. huge snapshot
2. some text-related conveniences are worse than in arXiv
3. using the API for all of science would be impractical

Conclusion:
**OpenAlex was selected as the best backbone of the project**

## 14.3 Semantic Scholar

### What it provides
1. Academic Graph API
2. papers, authors, citations, and venues
3. SPECTER2 embeddings
4. recommendations
5. a more AI-native semantic layer

### Limitations
1. less ideal as the main infrastructure for institutional metadata
2. more inconvenient practical rate limits for large-scale extraction
3. less appropriate than OpenAlex as the single reproducible backbone of the TFM

Conclusion:
OpenAlex is better as the foundation.
Semantic Scholar is useful as an additional layer for embeddings or semantic validation.

## 14.4 Google Scholar

Main conclusion:
**do not use it as the base of the project**
Reasons:
1. no comparable clean public API
2. difficult coverage control
3. embeddings not officially accessible
4. poorer reproducibility

---

# 15. OpenAlex vs arXiv

## For filtering fields or subfields
OpenAlex is better.

## For technical preprints with easy PDF access
arXiv is better.

## For institutions, countries, international collaboration, and rich metadata
OpenAlex is better.

## For basic text and fast setup within a STEM niche
arXiv may be more convenient.

Conclusion:
**use OpenAlex as the backbone**
**use arXiv as a complement when PDF or preprint access matters**

---

# 16. OpenAlex vs Semantic Scholar

## OpenAlex wins in
1. corpus definition
2. filters by field, subfield, and topic
3. institutions and countries
4. reproducibility
5. openness of the data

## Semantic Scholar wins in
1. scientific embeddings already available or more conceptually accessible
2. recommendations
3. modern semantic layer

Conclusion:
**OpenAlex first**
**Semantic Scholar as support if useful**

---

# 17. OpenAlex credits, API, and snapshot

## 17.1 API
The reviewed conclusion was:
for a TFM, the free limit is usually enough to extract a reasonable discipline or subfield if filtering is done properly.

## 17.2 Snapshot
The snapshot is **free**, but huge.
It was not recommended as the first option unless genuinely needed.

Operational conclusion:
1. use the API first
2. use the snapshot only if massive iteration is needed or the corpus grows too much

---

# 18. Discussed OpenAlex filters

Filters were proposed for the Statistics and Probability corpus:

## Reasonable initial filters
1. subfield = Statistics and Probability
2. has_abstract = true
3. type = article
4. language = en
5. has_references = true
6. is_retracted = false
7. is_paratext = false

## Optional filters depending on the objective
1. is_oa = true
2. has_pdf_url = true
3. has_fulltext = true

Recommendation:
build two corpus variants:
1. a broad base corpus without forcing open access
2. a stricter corpus for text-heavy experiments if needed

---

# 19. Discussion of the time variable

## What cannot be done well with arXiv alone
1. time to 100 citations
2. per-paper temporal accumulation of citations

arXiv alone provides well:
1. publication date in arXiv
2. update date
3. version information

## What can be done more robustly
1. citations at 3 years
2. log citations at 3 years
3. top 10 percent relative impact
4. maps by temporal windows
5. drift and regional growth

Conclusion:
**time should be treated mainly as field evolution and impact windows, not as exact time-to-event unless a suitable source appears**

---

# 20. Embeddings: decisions and resolved doubts

## 20.1 Do not train from scratch
Training a scientific embedding model from scratch is not advisable for the TFM.

## 20.2 Reasonable options
1. use own embeddings with an already-trained model
2. rely on SPECTER2
3. experiment with extended text, not only abstract, if feasible

## 20.3 Full text
It was discussed that using the full paper may be costly and noisy.
The most reasonable approach was:
1. baseline with abstract
2. optionally extended text or relevant sections

---

# 21. Conceptual web architecture

If the web is built, it should not try to become OpenAlex or a full Paperscape clone.
The viable TFM version would be:

1. reduced corpus
2. precomputed embeddings
3. precomputed coordinates or layout
4. fast rendering
5. filters by time, region, cluster, and maybe impact
6. static or nearly static hosting

## Suggested visual elements
1. distant view using density or hexagons
2. mid-level view using thematic regions
3. close view with individual papers
4. temporal layers
5. automatic region labels
6. possible impact relief layer

## What should NOT be done
1. do not rely on millions of raster tiles unless truly necessary
2. do not depend on a heavy backend
3. do not use a simple scatterplot as the final interface

---

# 22. Feasibility of a free public website

Conclusions:
1. yes, a public website for the project can be free or nearly free if traffic is low
2. the bottleneck is not so much the frontend, but the volume and format of the data
3. a modern architecture with compact data and browser-side rendering is better than copying the historical Paperscape system

Platforms considered viable:
1. Cloudflare Pages
2. GitHub Pages
3. Netlify in lightweight scenarios

---

# 23. Possible TFM titles

## Option 1
**Semantic and temporal cartography of scientific literature for thematic structure analysis and explainable prediction of future impact**

## Option 2
**Multi-scale cartography of a scientific subfield based on embeddings: representation, temporal evolution, and environmental variables**

## Option 3
**Geometry of scientific knowledge: construction of thematic maps and evaluation of their analytical utility**

## Option 4
**Map, environment, and impact: a cartographic and interpretable approach to scientific literature analysis**

---

# 24. Suggested next steps

## Phase 1. Finalize corpus
1. confirm subfield
2. define years
3. finalize filters
4. decide whether OA will be used as a restriction

## Phase 2. Build the master dataset
1. works
2. reconstructed abstract
3. authorships
4. institutions
5. countries
6. topics
7. cited_by_count and relative metrics if used

## Phase 3. Test representations
1. base embeddings
2. dimensionality reduction
3. clustering
4. visual aggregation
5. readability evaluation

## Phase 4. Temporal analysis
1. time windows
2. regions
3. growth
4. drift
5. impact relief

## Phase 5. Cartographic features
1. construct them per paper
2. normalize them
3. study collinearity
4. compare them against simple metadata

## Phase 6. Optional modeling
1. baseline
2. model with cartography
3. SHAP
4. conclusions on actual added value

## Phase 7. Web
1. only after the analytical core is sufficiently mature

---

# 25. Executive summary of the most important decisions

1. The TFM will be **a study**, not a website.
2. The website will be **a complementary prototype**.
3. OpenAlex will be the **backbone** of the project.
4. arXiv will be, at most, **a complementary source**.
5. Semantic Scholar is useful as support for embeddings or a semantic layer.
6. Prediction will not be the first objective.
7. SHAP is intended to close the interpretive argument.
8. The 2D representation cannot be a simple scatterplot.
9. The project must think in terms of **multi-scale structure, density, regions, and time**.
10. The discipline or corpus must be broad enough to show geography, but not so broad that it becomes noise.

---

# 26. Open points that are still unresolved

1. Confirm whether the final corpus will definitely be **Statistics and Probability** or whether a more visual macro-field will be reconsidered.
2. Decide the exact temporal window.
3. Decide whether the base corpus will include only articles or also proceedings.
4. Decide whether open access will be imposed in the base corpus or only in a subset.
5. Decide the exact embedding baseline.
6. Decide the main visual unit of the map.
7. Decide whether the predictive block remains in scope or is left as an optional final block.

---

# 27. Final operational recommendation for agents

If an agent continues this project, it should assume the following:

1. Do not redesign the project as a pure website.
2. Do not focus on prediction first.
3. Prioritize construction of a clean and well-defined corpus.
4. Test several representation and visual aggregation strategies.
5. Think of cartography as analytical structure, not just aesthetics.
6. Keep OpenAlex as the base source unless there is a strong reason not to.
7. Treat SHAP and explainability as the final validation block.
8. Document every decision about filters, coverage, and bias.

---

# 28. Message template for future agents

You can reuse this block when starting work with another agent:

```md
We are working on a master's thesis about semantic and temporal cartography of scientific literature.

This is not a pure web project. The core is methodological:
1. build a clean scientific corpus
2. represent papers in semantic space
3. convert that space into a readable multi-scale cartography
4. study regions, density, temporal drift, and possible environmental variables
5. optionally evaluate whether those variables help predict future impact
6. use SHAP or an equivalent tool to interpret whether cartographic features add real value

The main source currently chosen is OpenAlex.
The current corpus under consideration is the Statistics and Probability subfield with filters such as:
1. has_abstract = true
2. type = article
3. language = en
4. has_references = true
5. is_retracted = false
6. is_paratext = false

Important points:
1. avoid simple scatterplots as the final product
2. think in terms of multi-scale structure, regions, density, and time
3. the web is a complement, not the center
4. prediction is not the first step
5. document coverage biases and methodological decisions

We need help with [state the specific task].
```

---

# 29. Closing

The idea has evolved from "making a paper map" into something much stronger:

**using scientific cartography as an analytical instrument and, if it works, as a source of explainable variables to study future impact**

That shift is the key to the project.
