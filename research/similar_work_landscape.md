# Similar Work Landscape

Status: broad external scan of similar and adjacent work  
Last updated: 2026-04-24  
Purpose: assess whether the TFM idea is already saturated, partially covered, or still open enough to support a novel contribution

---

# 1. Executive verdict

The short answer is:

**the area is active, but your exact combination is not obviously "already done."**

What already exists is fragmented across four different traditions:

1. large science basemaps and bibliometric atlases
2. interactive literature discovery tools
3. semantic or hybrid clustering models for scientific corpora
4. impact prediction and AI-based literature synthesis

The strongest conclusion from this scan is that many people have built **maps**, many people have built **discovery tools**, and many people have built **impact predictors**, but I did **not** find a mature, standard line of work centered on:

1. building a semantic cartographic space for papers
2. extracting paper-level and region-level **cartographic variables**
3. tracking those regions **through time** as stable map objects
4. testing whether those cartographic variables improve explanation or prediction beyond text and metadata

That last combination still looks open enough to be interesting.

Important nuance:
this is an inference from the works reviewed below, not a proof that no one has ever tried something adjacent.

---

# 2. The landscape splits into four main families

## 2.1 Family A: Science basemaps and bibliometric atlases

This family tries to build a global or semi-global map of science itself.

Representative works:

1. [UCSD Map of Science](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0039464)
2. [Introducing the Open Biomedical Map of Science](https://www.frontiersin.org/articles/10.3389/frma.2023.1274793/full)
3. [Global Map of Science for OpenAlex 1.0](https://zenodo.org/records/16355791)

Core concepts:

1. build a reference basemap of disciplines or subdisciplines
2. classify journals or papers into those map regions
3. use overlays to show institutions, grants, topics, or countries on top of the basemap
4. optimize for high-level orientation and portfolio analysis

Why this matters for your TFM:

1. this is the strongest prior tradition for "science as a map"
2. it proves that mapping science is a real research area, not a gimmick
3. but most of these basemaps are about **global classification systems**, not paper-level semantic environments

Main gap relative to your idea:

1. the basemap tradition usually emphasizes classification and overlays
2. it less often treats the local cartographic neighborhood of each paper as a measurable object
3. it rarely closes the loop with explainable downstream modeling

## 2.2 Family B: Literature discovery and exploration tools

This family tries to help researchers find papers, not necessarily to study the geometry of a field scientifically.

Representative systems:

1. [Paperscape](https://paperscape.org/)
2. [Open Knowledge Maps](https://elifesciences.org/labs/ef274c83/open-knowledge-maps-a-visual-interface-to-the-world-s-scientific-knowledge)
3. [Litmaps](https://www.litmaps.com/)
4. [ResearchRabbit](https://learn.researchrabbit.ai/en/articles/12454660-how-does-researchrabbit-work)
5. [CitNetExplorer](https://www.citnetexplorer.nl/)
6. [VOSviewer](https://www.vosviewer.com/)

Core concepts:

1. start from a query or seed paper
2. use citation, reference, metadata, or text similarity
3. provide a graph, cluster view, or map to support exploration
4. help users discover relevant literature, influential papers, and topic neighborhoods

Why this matters for your TFM:

1. these tools are the closest product-level competitors to a "paper map" idea
2. they prove that visual exploration is useful to real users
3. they also show that the bar is already high if you want to compete on utility alone

Main gap relative to your idea:

1. most tools are about discovery workflows, not quantitative science-of-science questions
2. they usually do not turn map position into a reusable feature engineering framework
3. temporal evolution is often present as filters or timelines, not as formal region tracking
4. explanation and hypothesis testing are usually not the central goal

## 2.3 Family C: Semantic and hybrid modeling of scientific corpora

This family is closest to the analytic heart of your project.

Representative works:

1. [Semantic maps and metrics for science using deep transformer encoders](https://arxiv.org/abs/2104.05928)
2. [Mapping research topics at multiple levels of detail](https://www.sciencedirect.com/science/article/pii/S2666389921000209)
3. [Clustering More than Two Million Biomedical Publications](https://pmc.ncbi.nlm.nih.gov/articles/PMC3060097/)
4. [A detailed open access model of the PubMed literature](https://www.nature.com/articles/s41597-020-00749-y)
5. [Construction of the Literature Graph in Semantic Scholar](https://arxiv.org/abs/1805.02262)
6. [OpenAlex similar works / semantic search](https://docs.openalex.org/how-to-use-the-api/find-similar-works)

Core concepts:

1. use text, embeddings, citations, or hybrid signals to position documents
2. cluster papers into topics or research communities
3. build large scientific knowledge graphs and semantic search infrastructure
4. use hybrid relatedness measures when pure citation or pure text is not enough

Why this matters for your TFM:

1. this is where your project has the best methodological overlap
2. it validates the use of embeddings, hybrid similarity, and large open corpora
3. it suggests that semantic cartography is plausible and timely

Main gap relative to your idea:

1. many works stop at retrieval, clustering, or visual dashboards
2. fewer works operationalize a paper's map environment into explicit cartographic variables
3. fewer still connect those variables to temporal region dynamics and explainable prediction

## 2.4 Family D: Impact prediction and AI literature synthesis

This family is adjacent rather than identical, but it is important because it overlaps with your optional prediction layer.

Representative works:

1. [Prediction methods and applications in the science of science: A survey](https://www.sciencedirect.com/science/article/pii/S1574013719300759)
2. [A review of scientific impact prediction: tasks, features and methods](https://ideas.repec.org/a/spr/scient/v128y2023i1d10.1007_s11192-022-04547-8.html)
3. [Realistic Citation Count Prediction Task for Newly Published Papers](https://aclanthology.org/2023.findings-eacl.84/)
4. [OpenScholar](https://www.nature.com/articles/s41586-025-10072-4)
5. [Elicit Systematic Literature Reviews](https://elicit.com/solutions/literature-review)

Core concepts:

1. predict future scientific impact or citation trajectories
2. automate evidence synthesis and literature review steps
3. use retrieval, structured screening, extraction, and citation-backed generation
4. validate whether predictions or syntheses are realistic and transparent

Why this matters for your TFM:

1. it shows that the broader space of scientific literature intelligence is crowded
2. if you build only "an AI tool to review literature," you enter a very competitive area
3. your stronger angle is not generic review automation, but quantitative cartographic analysis

Main gap relative to your idea:

1. these systems do not usually center the map itself as the scientific object of study
2. they often lack explicit region-level cartography
3. they do not typically ask whether a paper's local semantic landscape adds independent predictive value

---

# 3. Closest similar works, one by one

## 3.1 [Paperscape](https://paperscape.org/)

What it is:

1. an interactive map of arXiv papers
2. each paper is a circle
3. size reflects citation impact
4. position comes from a force-style layout based on references

Why it feels close to your idea:

1. it has the strongest "atlas" feeling of the public tools
2. it shows papers as geography rather than only lists or citation chains
3. it demonstrates the communication power of a territorial metaphor

Why it is not the same as your TFM:

1. it is arXiv-centered rather than corpus-method centered
2. it is mainly a navigation and visual exploration system
3. it does not frame the map as a source of cartographic variables for analysis
4. it does not provide a strong temporal region-tracking or explainability agenda

Main concept to keep:

1. separate layout from final rendering
2. think in atlas terms, not scatterplot terms
3. multi-scale navigation matters

## 3.2 [Open Knowledge Maps](https://elifesciences.org/labs/ef274c83/open-knowledge-maps-a-visual-interface-to-the-world-s-scientific-knowledge)

What it is:

1. an open, query-driven knowledge mapping system
2. it creates clusters from text and metadata
3. it labels clusters automatically
4. it follows "overview first, zoom and filter, then details-on-demand"

Why it feels close:

1. it directly visualizes literature around a topic
2. it produces interpretable thematic clusters
3. it already combines retrieval, clustering, labeling, and interactive exploration

Why it is still different:

1. the main goal is discovery support, not science-of-science measurement
2. the map is generated per query, not necessarily as a stable field-wide semantic coordinate system
3. it does not focus on temporal cartography or predictive evaluation

Main concept to keep:

1. automatic labels matter
2. map interaction should support human interpretation
3. open infrastructure is a meaningful differentiator

## 3.3 [Litmaps](https://www.litmaps.com/) and [ResearchRabbit](https://learn.researchrabbit.ai/en/articles/12454660-how-does-researchrabbit-work)

What they are:

1. seed-paper-based literature discovery tools
2. both emphasize citation-network exploration
3. both help users expand from known papers to related ones
4. ResearchRabbit additionally stresses timelines, topic evolution, and collection workflows

Why they feel close:

1. they make literature look like a connected landscape
2. they support "find the neighborhood around a paper"
3. they are probably the closest popular tools to what many users imagine when they say "map the literature"

Why they are still different:

1. they optimize for workflow utility, not a defensible thesis contribution
2. they do not turn local map structure into a formal feature set
3. their temporal component is mainly navigational rather than region-analytic

Main concept to keep:

1. seed-based exploration is powerful
2. users like seeing how papers connect across time
3. a useful demo can exist without being the thesis core

## 3.4 [VOSviewer](https://www.vosviewer.com/) and [CitNetExplorer](https://www.citnetexplorer.nl/)

What they are:

1. mature scientometric tools for citation, coupling, co-occurrence, and citation-network analysis
2. widely used in bibliometric studies
3. strong baselines for mapping and network exploration

Why they matter:

1. if your project only recreates standard VOSviewer or CitNetExplorer workflows, it will feel derivative
2. reviewers may ask why custom code is needed if a standard scientometric tool gives similar outputs

Why they are still different from your idea:

1. they are optimized for bibliometric network analysis, not semantic cartographic feature engineering
2. they do not naturally frame the problem as "does local semantic environment predict or explain impact?"
3. their maps are typically the end product, not an input to downstream interpretable modeling

Main concept to keep:

1. always compare your outputs to strong bibliometric baselines
2. your innovation must be clearer than "I made another map"

## 3.5 [UCSD Map of Science](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0039464)

What it is:

1. one of the canonical basemaps of science
2. designed as a general classification system and global reference map
3. used for overlays of institutions, competencies, and portfolios

Why it matters:

1. it proves the legitimacy of science cartography
2. it gives a baseline model of what a "map of science" means in practice
3. it emphasizes coverage, classification design, and overlay usability

Why it is not your project:

1. it is not centered on paper-level semantic neighborhoods
2. it is not centered on temporal drift at the region level
3. it is not centered on downstream explainable prediction

Main concept to keep:

1. basemaps are useful reference systems
2. a map becomes much stronger when it supports overlays and comparison

## 3.6 [Open Biomedical Map of Science](https://www.frontiersin.org/articles/10.3389/frma.2023.1274793/full)

What it is:

1. an effort to build an open, fair biomedical basemap from PubMed and MeSH-linked structure
2. explicitly motivated by reproducibility and FAIR principles
3. partly a response to closed commercial basemaps

Why it matters:

1. it shows that open-data science mapping is becoming a serious agenda
2. it narrows the novelty of "I will use open data to build a science map"

Why it is still different:

1. the focus is basemap construction and openness
2. not the extraction of cartographic variables from a semantic paper landscape
3. not the paper-level explainability problem

Main concept to keep:

1. openness and reproducibility are real contributions, but not enough alone

## 3.7 [Global Map of Science for OpenAlex 1.0](https://zenodo.org/records/16355791)

What it is:

1. a recent OpenAlex-based global map release
2. evidence that map-of-science infrastructure is already moving onto open scholarly graphs

Why it matters:

1. it reduces the novelty of "OpenAlex-backed science map" as a standalone claim
2. it increases the importance of your methodological angle

Why it is still different:

1. it appears to be an infrastructure or basemap resource, not a thesis about semantic local environment and cartographic features
2. it does not obviously occupy the temporal-analysis plus explainability position you are considering

Main concept to keep:

1. OpenAlex is now strong enough that many map projects will converge on it
2. you need a claim stronger than "I mapped OpenAlex"

## 3.8 [Semantic maps and metrics for science using deep transformer encoders](https://arxiv.org/abs/2104.05928)

What it is:

1. a close conceptual neighbor to your semantic cartography idea
2. it argues that deep contextual scientific representations improve mapping science
3. it explicitly links embeddings to understanding and forecasting research communities

Why it matters:

1. this is one of the closest academic overlaps with your semantic direction
2. it narrows the novelty of "use transformers to map science"

Why it is still different:

1. it is about better semantic representations and metrics
2. it does not fully spell out the cartographic-feature-engineering thesis you are building
3. it does not appear to be centered on temporal atlas design plus SHAP-style evaluation

Main concept to keep:

1. semantic science mapping with transformers is real and credible
2. your novelty must come from what you *do with the map*, not just how you embed papers

## 3.9 [Mapping research topics at multiple levels of detail](https://www.sciencedirect.com/science/article/pii/S2666389921000209)

What it is:

1. a topic-mapping workflow for institutional review
2. interactive multilevel maps
3. temporal snapshots and different granularity levels

Why it matters:

1. it shows that multiscale research maps with time filters already exist in academic form
2. it narrows the novelty of "interactive multilevel topic map"

Why it is still different:

1. it is topic-centered rather than paper-cartography-centered
2. it is designed for institutional review support
3. it does not focus on local paper environment, cartographic variables, or predictive ablation

Main concept to keep:

1. multi-level detail is useful
2. temporal snapshots are not enough by themselves; they should support real analytical questions

## 3.10 [Clustering More than Two Million Biomedical Publications](https://pmc.ncbi.nlm.nih.gov/articles/PMC3060097/) and [A detailed open access model of the PubMed literature](https://www.nature.com/articles/s41597-020-00749-y)

What they are:

1. large-scale biomedical literature modeling efforts
2. heavy emphasis on clustering quality and relatedness measures
3. the later PubMed model combines direct citation and text similarity in a 50:50 hybrid measure

Why they matter:

1. they prove that large hybrid scientific clustering is already possible at scale
2. they narrow the novelty of "large open corpus + clustering"

Why they are still different:

1. they are more about cluster construction and large-scale modeling than cartographic interpretation
2. they do not center the region-drift or paper-level-feature problem

Main concept to keep:

1. hybrid text-plus-citation relatedness is likely better than either alone
2. large-scale open corpora are tractable

## 3.11 [Semantic Scholar literature graph](https://arxiv.org/abs/1805.02262) and [OpenAlex semantic search](https://docs.openalex.org/how-to-use-the-api/find-similar-works)

What they are:

1. infrastructure layers for scientific discovery
2. large heterogeneous graphs or large semantic indexes
3. retrieval by citation, metadata, entities, or embeddings

Why they matter:

1. they mean basic scientific discovery infrastructure is already highly developed
2. your project should not position itself as "I built a graph of papers" and stop there

Why they are still different:

1. they are infrastructure, not necessarily map-centered analysis
2. they expose building blocks that your TFM can use rather than directly replace

Main concept to keep:

1. you can lean on these infrastructures instead of reinventing them
2. the contribution should sit above the infrastructure layer

## 3.12 [OpenScholar](https://www.nature.com/articles/s41586-025-10072-4) and [Elicit](https://elicit.com/solutions/literature-review)

What they are:

1. systems for scientific literature synthesis rather than science cartography
2. retrieval-augmented or interactive evidence-review systems
3. strong competition in "AI helps with literature review"

Why they matter:

1. they make the generic literature-review-assistant space very crowded
2. they raise user expectations around citation-backed synthesis, screening, and extraction

Why they are still different:

1. they do not make the map the object of study
2. they do not usually measure region drift, local density, or semantic frontier behavior
3. they do not test cartographic feature families as explanatory variables

Main concept to keep:

1. if you want a demo, position it as a complement to cartographic analysis, not a generic AI reviewer

---

# 4. What appears already done

Based on the reviewed works, the following directions already look crowded or routine:

## 4.1 "Make a visual graph of related papers"

Already covered by:

1. Paperscape
2. ResearchRabbit
3. Litmaps
4. CitNetExplorer
5. many VOSviewer-style bibliometric analyses

## 4.2 "Build a science map from citation or co-occurrence data"

Already covered by:

1. UCSD Map of Science
2. VOSviewer
3. CiteSpace
4. bibliometrix
5. Open Biomedical Map of Science
6. OpenAlex/Open map releases

## 4.3 "Use embeddings or modern NLP to find similar papers"

Already covered by:

1. Semantic Scholar literature graph and retrieval stack
2. OpenAlex semantic search
3. transformer-based semantic science mapping papers
4. SPECTER/SPECTER2-style scientific embeddings from the literature review file

## 4.4 "Use AI to summarize scientific literature"

Already covered by:

1. OpenScholar
2. Elicit
3. many commercial or semi-commercial literature assistants

## 4.5 "Predict citations from paper metadata or text"

Already covered by:

1. the science-of-science prediction surveys
2. multiple neural and graph-based citation prediction papers
3. realistic citation benchmark work

---

# 5. What still looks open or only partially covered

This is the part that matters most for your novelty.

## 5.1 The map as a feature generator

This still looks promising.

What seems missing:

1. a clean definition of paper-level cartographic variables such as density, boundary distance, centrality-versus-periphery, local heterogeneity, local novelty, and local temporal change
2. a systematic test of whether these variables add signal beyond text and metadata
3. a feature-family perspective where "cartography" becomes a measurable modeling block

Why it matters:
most prior systems stop at map creation or navigation. They do not fully operationalize the map as structured input for analysis.

## 5.2 Temporally stable semantic atlas design

This also looks promising.

What seems missing:

1. region identities tracked through time in a stable semantic coordinate system
2. formal metrics for region birth, drift, fragmentation, consolidation, and frontier expansion
3. a convincing bridge between temporal science mapping and semantic embeddings

Why it matters:
many systems have time filters, timelines, or year slices, but fewer treat temporal cartography itself as the research question.

## 5.3 Cartographic explanation, not just prediction

This is probably one of your strongest openings.

What seems missing:

1. SHAP-style or interpretable analysis grouped by feature families where cartographic features are explicitly tested
2. evidence that local environment matters more than global topic label
3. explanation of *why* certain map positions lead to later outcomes

Why it matters:
explainability exists in ML, and citation prediction exists in scientometrics, but their intersection with semantic cartography is much less standardized.

## 5.4 A reproducible open-corpus pipeline

This is not enough by itself, but it strengthens the thesis.

What seems missing:

1. a full open pipeline built on OpenAlex and/or PubMed-like open sources
2. reproducible semantic and temporal cartography that others can rerun
3. open outputs that are analytically richer than standard VOSviewer dashboards

Why it matters:
open basemaps now exist, but there is still room for reproducible *analysis-first* cartography.

---

# 6. The strongest novelty positions for your TFM

If you want the project to feel clearly differentiated, these are the best positions.

## 6.1 Best position: cartography as measurable scientific context

Proposed claim:

1. the map is not just a visualization
2. it is a coordinate system from which measurable contextual variables can be extracted
3. those variables reveal scientific structure and possibly predictive signal

Why this is strong:

1. it is more analytical than Paperscape, Litmaps, or ResearchRabbit
2. it is more semantic and paper-level than classical basemaps
3. it gives a thesis contribution even if the web demo is modest

## 6.2 Best position: temporal region dynamics in semantic space

Proposed claim:

1. scientific regions can be tracked over time in a semantic atlas
2. drift, growth, fragmentation, and emergence can be quantified
3. these dynamics reveal frontier behavior invisible in lists of papers

Why this is strong:

1. it gives you a strong non-predictive thesis core
2. it avoids competing directly with discovery tools
3. it aligns well with your current idea bank

## 6.3 Best position: do cartographic features add value?

Proposed claim:

1. compare text + metadata baselines against text + metadata + cartography
2. analyze whether local semantic environment improves explanation or prediction
3. use interpretable models to identify which cartographic families matter

Why this is strong:

1. it creates a falsifiable research question
2. it avoids vague novelty claims
3. even a negative result is interesting if the cartographic block fails clearly and informatively

## 6.4 Good supporting position: open, reproducible, medium-scale field atlas

Proposed claim:

1. use an open corpus
2. target a coherent macro-field rather than all of science
3. build a reusable atlas pipeline that others can inspect and rerun

Why this helps:

1. it gives you practical feasibility
2. it avoids the impossible scope of mapping all science at once
3. it still produces a meaningful demo

---

# 7. Directions that now look weak or too crowded

These are the directions I would avoid making the central thesis claim.

## 7.1 "A website like Paperscape for my field"

Why weak:

1. too close to existing discovery tools
2. hard to defend academically
3. likely to become a frontend-heavy project

## 7.2 "A UMAP scatterplot of paper embeddings"

Why weak:

1. this is now routine
2. it is visually unstable
3. it does not by itself produce a scientific contribution

## 7.3 "A standard bibliometric review using VOSviewer/CiteSpace"

Why weak:

1. extremely common
2. low novelty unless the domain itself is special
3. not well aligned with your strongest idea

## 7.4 "Generic AI literature review assistant"

Why weak:

1. OpenScholar and Elicit already make this a highly contested space
2. it would move you away from your most distinctive cartographic angle

## 7.5 "Raw citation prediction"

Why weak:

1. heavily studied
2. easy to do badly because of leakage and age effects
3. not unique without the cartographic contribution

---

# 8. My current recommendation

If I had to choose the cleanest novelty statement right now, it would be:

**Build a reproducible semantic and temporal atlas of a coherent scientific field, define paper-level and region-level cartographic variables from that atlas, and test whether those variables reveal or explain scientific structure beyond text, metadata, and standard topic labels.**

If you want the optional predictive block, then the strongest extension is:

**evaluate whether cartographic features improve realistic, time-aware impact prediction, and explain the result with feature-family-level interpretability.**

That still looks differentiated.

---

# 9. Bottom line

The space is **not empty**, but it is also **not closed**.

My best reading of the current landscape is:

1. the "map as interface" idea is crowded
2. the "map as bibliometric basemap" idea is mature
3. the "AI summarizes literature" idea is crowded fast
4. the "map as measurable semantic context with temporal dynamics and interpretable evaluation" idea still looks fresh enough for a strong TFM

So the project should not try to beat every existing literature tool at discovery.
It should try to answer a sharper research question that these tools mostly do not ask.

---

# 10. Sources used in this scan

Core systems and infrastructures:

1. [Paperscape](https://paperscape.org/)
2. [Open Knowledge Maps / Head Start architecture](https://elifesciences.org/labs/ef274c83/open-knowledge-maps-a-visual-interface-to-the-world-s-scientific-knowledge)
3. [Litmaps](https://www.litmaps.com/)
4. [ResearchRabbit algorithm overview](https://learn.researchrabbit.ai/en/articles/12454660-how-does-researchrabbit-work)
5. [VOSviewer](https://www.vosviewer.com/)
6. [CitNetExplorer](https://www.citnetexplorer.nl/)
7. [OpenAlex similar works](https://docs.openalex.org/how-to-use-the-api/find-similar-works)
8. [OpenAlex as a map of the research ecosystem](https://help.openalex.org/hc/en-us/articles/28932712154391-How-does-OpenAlex-work)
9. [Semantic Scholar literature graph](https://arxiv.org/abs/1805.02262)

Core academic works:

1. [UCSD Map of Science](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0039464)
2. [Open Biomedical Map of Science](https://www.frontiersin.org/articles/10.3389/frma.2023.1274793/full)
3. [Global Map of Science for OpenAlex 1.0](https://zenodo.org/records/16355791)
4. [Semantic maps and metrics for science using deep transformer encoders](https://arxiv.org/abs/2104.05928)
5. [Mapping research topics at multiple levels of detail](https://www.sciencedirect.com/science/article/pii/S2666389921000209)
6. [Clustering More than Two Million Biomedical Publications](https://pmc.ncbi.nlm.nih.gov/articles/PMC3060097/)
7. [A detailed open access model of the PubMed literature](https://www.nature.com/articles/s41597-020-00749-y)
8. [Prediction methods and applications in the science of science: A survey](https://www.sciencedirect.com/science/article/pii/S1574013719300759)
9. [A review of scientific impact prediction: tasks, features and methods](https://ideas.repec.org/a/spr/scient/v128y2023i1d10.1007_s11192-022-04547-8.html)
10. [Realistic Citation Count Prediction Task for Newly Published Papers](https://aclanthology.org/2023.findings-eacl.84/)
11. [OpenScholar](https://www.nature.com/articles/s41586-025-10072-4)
12. [Elicit systematic reviews](https://elicit.com/solutions/literature-review)
