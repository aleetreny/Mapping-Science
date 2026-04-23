# Ideas Database
## Structured bank of concepts, experiments, and possible directions for the TFM

This file is intentionally exploratory.
It should collect:

1. project variants
2. methodological alternatives
3. literature inspired ideas
4. visualization concepts
5. feature engineering ideas
6. predictive extensions
7. questions worth testing

---

# 1. Project variants considered

## 1.1 Pure cartographic atlas

Goal:
Build a navigable scientific atlas with regions, zoom, labels, and temporal layers.

Potential value:

1. useful visual exploration of a field
2. strong communication value
3. useful demo for thesis defense and portfolio

Risk:

1. may be scientifically weak if not tied to a concrete research question

## 1.2 Cartography as analytical representation

Goal:
Treat the map as a representation from which measurable region level and paper level properties can be extracted.

Potential value:

1. stronger thesis contribution
2. easier to validate
3. compatible with later modeling

## 1.3 Cartography plus explainable prediction

Goal:
Use cartographic features together with text and metadata to predict future impact.

Potential value:

1. closes the loop from representation to explanation
2. allows SHAP based interpretation

Risk:

1. predictive signal may be weak or unstable
2. should not become the only pillar of the thesis

---

# 2. Candidate research questions

## 2.1 Representation centered

1. Can a multi scale cartographic representation capture interpretable thematic structure better than a simple 2D projection?
2. Which aggregation strategies make a scientific map readable?
3. Can local cartographic structure reveal field organization that is hidden in raw point clouds?

## 2.2 Temporal centered

1. How do thematic regions evolve over time?
2. Can drift, growth, and fragmentation of scientific regions be measured cartographically?
3. What temporal structures emerge when the field is observed as a landscape rather than a list of papers?

## 2.3 Prediction centered

1. Do cartographic features improve impact prediction beyond text and metadata?
2. Which cartographic variables matter most if the model succeeds?
3. Is local environment more informative than global topic label?

---

# 3. Visualization concepts

## 3.1 Hexagonal map

Represent the scientific space with hexagonal bins.

Advantages:

1. reduces clutter
2. readable at multiple scales
3. supports density and thematic summaries

Possible variables per hexagon:

1. dominant topic
2. mean publication year
3. mean citations
4. growth rate
5. representative paper

## 3.2 Continuous density map

Represent the field as a smoothed density surface.

Possible layers:

1. density of papers
2. recency heatmap
3. impact relief
4. novelty hotspots

## 3.3 Region based thematic atlas

Use clustering or community detection to define regions.

Possible design:

1. regions with labels
2. subregions revealed on zoom
3. paper level view only when close enough

## 3.4 Temporal atlas

Keep the same base space but show:

1. yearly snapshots
2. region transitions
3. growth and shrinkage
4. moving frontiers

## 3.5 Landscape metaphor

Use relief language explicitly.

Possible landscape variables:

1. paper density as elevation
2. citations as elevation
3. novelty as heat
4. temporal expansion as contour change

---

# 4. Candidate cartographic features by paper

## 4.1 Density features

1. local semantic density
2. neighbor count within radius
3. kernel density estimate
4. density percentile within cohort

## 4.2 Position features

1. distance to cluster centroid
2. distance to nearest boundary
3. central versus peripheral score
4. local anisotropy of neighborhood

## 4.3 Heterogeneity features

1. topic entropy of local neighborhood
2. institutional diversity in local neighborhood
3. country diversity in local neighborhood
4. semantic spread of local neighbors

## 4.4 Novelty features

1. distance to previous year neighborhood
2. rarity relative to historical papers
3. uncommon mixture score
4. semantic deviation from dominant local topic

## 4.5 Relief features

1. local citation mean
2. citation gradient
3. citation ruggedness
4. local citation rank among neighbors

## 4.6 Temporal features

1. neighborhood growth in recent window
2. change in local density
3. local centroid movement
4. fragmentation or consolidation index
5. survival of neighboring micro clusters

## 4.7 Hybrid graph features

1. similarity graph degree
2. centrality in similarity graph
3. bridge score between neighboring communities
4. overlap between semantic and citation neighborhoods

---

# 5. Candidate predictive targets

## 5.1 Safer targets

1. log citations after 3 years
2. top 10 percent impact in same cohort and subfield
3. top 20 percent impact in same cohort and subfield
4. citation percentile style target

## 5.2 Harder or less recommended targets

1. exact time to 100 citations
2. exact long horizon raw citation count
3. broad cross field impact comparison without normalization

---

# 6. Candidate model families

## 6.1 Interpretable tabular models

1. logistic regression
2. random forest
3. XGBoost
4. CatBoost

Use case:

1. strong baselines
2. easy SHAP integration
3. suitable for feature family comparison

## 6.2 Neural models

1. MLP on tabular features
2. model using paper embeddings plus engineered features

Use case:

1. benchmark only
2. not preferred as main thesis model if interpretability is central

---

# 7. Candidate domain scopes

## 7.1 Narrow niche

Examples:

1. computer vision
2. NLP
3. reinforcement learning

Pros:

1. cleaner semantics
2. more coherent vocabulary

Cons:

1. may be too narrow for a geographic effect

## 7.2 Intermediate macro field

Examples:

1. neural networks
2. AI and data science related subfields
3. statistics and probability

Pros:

1. enough diversity for visible regions
2. still coherent enough to be interpretable

## 7.3 Extremely broad corpus

Examples:

1. all of science
2. all STEM

Cons:

1. too heterogeneous
2. likely too noisy for a first TFM version

---

# 8. Data source strategy ideas

## 8.1 OpenAlex as backbone

Best for:

1. corpus definition
2. institutions and countries
3. citations and metadata
4. topic filtering

## 8.2 arXiv as complement

Best for:

1. technical preprints
2. PDF access
3. STEM repository categories

## 8.3 Semantic Scholar as semantic support

Best for:

1. SPECTER2 based support
2. recommendations
3. similarity checks

---

# 9. Possible experiment blocks

## 9.1 Representation experiments

1. compare multiple embedding baselines
2. compare multiple 2D projection strategies
3. compare clustering methods
4. compare visual aggregation strategies

## 9.2 Temporal experiments

1. compare static map versus sliding windows
2. measure region drift
3. identify stable versus unstable regions
4. compare early and late field structure

## 9.3 Predictive experiments

1. text plus metadata baseline
2. text plus metadata plus cartography
3. cartography only versus metadata only
4. ablation by feature family

## 9.4 Explainability experiments

1. SHAP global ranking
2. SHAP by feature family
3. SHAP local case studies for representative papers
4. compare explanation profiles in different thematic regions

---

# 10. High value outputs for the thesis

1. a clean corpus definition
2. a readable multi scale map
3. quantitative evidence of thematic regions and temporal evolution
4. a reusable feature engineering framework for cartographic variables
5. an interpretable evaluation of whether cartography adds value
6. a web demonstrator if time allows

---

# 11. Questions still worth exploring

1. Should the final corpus stay in Statistics and Probability or move to a broader macro field?
2. What visual unit is best for the final map: points, hexagons, density, or regions?
3. Which cartographic variables are robust enough to survive different embedding and layout choices?
4. Can the temporal layer become a strong enough thesis contribution even if prediction underperforms?
5. If the predictive block fails, what is the strongest non predictive version of the thesis?

---

# 12. Agent usage note

Future agents should use this file to:

1. expand promising ideas
2. record literature inspired directions
3. mark discarded options clearly
4. prevent the project from collapsing into a single narrow implementation path
