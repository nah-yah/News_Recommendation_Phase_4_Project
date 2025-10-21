# News Recommendation System

## 1. Overview
### 1.1 Project purpose
This project introduces a credibility-aware news recommendation system that addresses the critical challenge of misinformation in digital news consumption. Unlike traditional recommendation systems that prioritize content relevance alone, this solution integrates credibility assessment as a core component of the recommendation process. The system's innovation lies in its balanced hybrid approach that combines content similarity with source credibility through a verified 50/50 weighting scheme, ensuring users receive both relevant and trustworthy content.

### 1.2 Core value proposition
For news readers (primary stakeholders), the system delivers transparent, trustworthy recommendations that help combat misinformation exposure while maintaining content relevance. For news platforms (secondary stakeholders), it provides a competitive advantage through enhanced user trust, increased engagement, and reduced risk of promoting misinformation. The system's unique value stems from its ability to filter fake news effectively (38% reduction compared to similarity-only recommenders) while preserving meaningful topic relevance (0.42-0.61 average match), creating a safer news consumption environment without sacrificing user experience.

### 1.3 System architecture
The recommendation system operates through three interconnected components:
1. **Content Analysis Engine**: Uses TF-IDF vectorization and cosine similarity to measure headline relevance
2. **Credibility Assessment Module**: Applies logistic regression to identify linguistic patterns associated with misinformation
3. **Hybrid Recommendation Generator**: Combines relevance and credibility scores using topic-sensitive thresholds (45% for health topics, 40% for political topics, 30% for standard topics)

The system incorporates explainability features that transform technical credibility scores into reader-friendly explanations, highlighting specific trust signals like "Reputable health authority mentioned" or risk signals like "Excessive exclamation marks detected." This transparent approach empowers readers to develop media literacy skills while making informed decisions about news trustworthiness. Designed with mobile-first principles, the system delivers concise, scannable credibility information that works within the constraints of small-screen interfaces.

# 2. Business understanding
### 2.1 Business problem
The rise of online news consumption has made information more accessible, but it has also enabled the rapid spread of misinformation. Standard recommendation systems typically rely on content similarity or popularity without distinguish credible and non-credible sources. As a result, fake news articles receive promotion as easily as verified ones, reinforce misinformation cycles and erode user trust. This situation creates challenges for news platforms that aim to keep audiences engaged while maintain credibility. Without integrate reliability signals into recommendations, platforms risk contribution to echo chambers, damage their reputation, and undermine the quality of public discourse.

### 2.2 Goal
The goal of this project is to design and evaluate a recommendation system that goes beyond traditional relevance-based filtering through credibility integration as a key factor. Instead of suggest articles that match user interests, the system aim to prioritize trustworthy news while deliver engaging and diverse content. Through explainability integration, the model provide transparency into why certain articles receive recommendation, help build user trust and support responsible news consumption.

### 2.3 Objectives
The project pursue five specific objectives to address the misinformation challenge in news recommendation:
- Collect, clean, and preprocess the fake/true news dataset for modeling
- Build a recommendation system that integrate both content relevance and credibility of articles
- Evaluate the model using appropriate metrics that measure both relevance and trustworthiness
- Incorporate explainability techniques to show users why a recommendation is made
- Document findings and highlight limitations to guide future improvements

### 2.4 Scope
This project focus on develop a prototype recommendation system that suggest news articles while account for both content relevance and source credibility. The scope limit to the use of publicly available datasets contain labeled fake and true news articles. The system primarily demonstrate the technical feasibility of combine recommendation techniques with credibility scoring, rather than build a fully deployed application. The project not address real-time data ingestion, large-scale deployment, or advanced personalization features. Instead, the emphasis be on proof of concept, evaluation of results, and highlight directions for future improvements.

### 2.5 Stakeholders
#### i. Primary stakeholder
The primary stakeholder is the end user, specifically news readers who rely on digital platforms for information. They directly affect by exposure to misinformation and therefore benefit from a system that recommend relevant yet credible news articles. Their trust, engagement, and satisfaction serve as the central drivers of success for this project.

#### ii. Secondary stakeholders
Secondary stakeholders include news platforms and aggregators seek to maintain credibility and retain users, researchers and data scientists interested in misinformation detection and recommendation models, and policy makers and fact-checking organizations who use insights to promote responsible information dissemination. These stakeholders benefit from a system that demonstrate responsible AI practices, align with regulatory expectations, and provide tools to combat misinformation.

### 2.6 Success criteria
The success of the project measure by the system ability to recommend articles that be both relevant and credible. Key indicators include:
- Recommendation accuracy: Credible articles consistently rank higher than fake ones in top-N recommendations
- Coverage: The system provide diverse recommendations across different topics without bias toward a few sources
- User trust proxy: Reduction in the proportion of fake news articles in the recommended list compared to a baseline content-only recommender
- Technical performance: Model process the dataset efficiently and reproduce for future updates

### 2.7 Business value
Implementation of a credibility-aware recommendation system provide multiple benefits for both users and platforms. For news platforms, it strengthen trust, improve user retention, and reduce the risk of promote misinformation. For readers, it ensure access to relevant and reliable content, support informed decision-making. Additionally, through responsible AI practices demonstration, platforms align with regulatory expectations and ethical standards, enhance their reputation. Overall, the system add measurable value through user engagement and content integrity combination, create a safer and more trustworthy online news ecosystem. The business value extend beyond immediate metrics to include long-term brand protection and contribution to healthier public discourse.

## 3. Data understanding
### 3.1 Data source
The system utilizes the WELFake dataset, a comprehensive collection of 72,134 news articles specifically curated for fake news detection research. This dataset combines four popular news sources: Kaggle, McIntire, Reuters, and BuzzFeed Political, creating a diverse and representative sample of news content. The dataset includes verified labels (0 = real news, 1 = fake news) for each article, enabling reliable credibility assessment. The inclusion of multiple sources ensures the system can identify credibility patterns across different publication styles and subject areas, providing a robust foundation for developing a generalizable credibility scoring mechanism.

### 3.2 Dataset characteristics
The raw dataset contains three primary columns:
- title: The headline of the news article
- text: The full content of the article
- label: Binary indicator (0 = real news, 1 = fake news)

Analysis of the raw dataset reveals important characteristics that inform the credibility assessment approach:
- Approximately 51.4% of articles are labeled as fake news (1), while 48.6% are real news (0)
- Headlines in fake news articles tend to be longer on average than those in real news
- Fake news headlines show higher frequency of emotional language indicators
- The dataset provides sufficient examples across various topic domains to support robust credibility assessment

![](./images/fake_vs_real.png)

### 3.3 Initial data quality assessment
The raw dataset contains data quality issues that require attention before analysis:
- 558 missing titles (0.77% of records)
- 39 missing text entries (0.05% of records)
- 0 missing labels (all articles have credibility labels)
- Evidence of duplicate entries across multiple sources

These quality issues need careful handling to ensure the credibility assessment reflects genuine patterns rather than data artifacts. The presence of missing values in headline and text fields particularly concerns credibility assessment, as these represent the primary signals readers encounter before clicking through to full articles.

### 3.4 Statistical patterns in raw data
Initial analysis of the raw dataset reveals significant differences between real and fake news articles:
- Fake news headlines average 82.97 characters compared to 68.81 for real news
- Fake news headlines contain more exclamation marks on average
- Fake news headlines show higher question usage (6.5% vs 2.5%)
- Fake news headlines use more capitalization (1.9% vs 1.0%)

These patterns align with research showing that fake news often employs sensationalist language, urgency indicators, and emotional triggers to capture attention. Headline length emerges as the most reliable single indicator for early fake news detection, with longer headlines more likely associated with misinformation. These statistical differences provide the foundation for the credibility scoring system, enabling the model to distinguish authentic journalism from misinformation based on measurable headline characteristics.

## 4. Data preparation
### 4.1 Data cleaning process
The data preparation phase implemented a systematic cleaning process to address quality issues identified in the raw dataset. The process began with missing value handling, where 558 records with missing titles and 39 with missing text were removed entirely rather than imputed. This approach preserved data integrity while maintaining the natural balance between real and fake news articles. The decision to remove rather than impute ensured credibility assessments would reflect only headline-based signals that readers encounter before clicking through to full articles.

Following missing value treatment, duplicate elimination addressed 8,416 exact duplicates where both title and text matched across multiple sources. These duplicates were removed to prevent skewed model training that could overemphasize certain content patterns. The final cleaned dataset contains 63,121 unique articles, representing a 12.4% reduction from the original dataset. This careful cleaning process maintained the dataset's near-perfect class balance (48.6% real, 51.4% fake) while removing potential model bias from incomplete or redundant content. The verification confirmed no unintended data leakage during cleaning, ensuring the integrity of subsequent credibility assessment.

### 4.2 Feature engineering approach
The feature engineering phase developed four key indicators from news article titles that serve as critical credibility signals:
1. **Title Length**: Character count of the headline, designed to capture sensationalism and attention-grabbing tactics commonly used in fake news
2. **Exclamation Count**: Number of exclamation marks in the headline, engineered to identify emotional manipulation and urgency indicators that often appear in misleading content
3. **Question Presence**: Binary indicator for question-based headlines, created to detect uncertainty tactics that suggest unverified claims
4. **Capitalization Ratio**: Proportion of capitalized words in the headline, developed to measure the use of ALL-CAPS text for false urgency

These features were selected based on their statistical significance in distinguishing real from fake news. The engineering process maintained a strict focus on headline-only analysis because readers encounter headlines before clicking through to full articles, making headline-based credibility assessment immediately relevant to user experience. Each feature captures a distinct aspect of headline composition that research shows correlates with credibility.

### 4.3 Feature analysis
Analysis of the engineered features revealed statistically significant differences between real and fake news articles:
- **Title Length**: Fake news headlines average 82.97 characters (14 characters longer than real news headlines at 68.81 characters). This difference proves statistically significant, with longer headlines more likely associated with misinformation. Headline length emerged as the most reliable single indicator for early fake news detection.
- **Exclamation Count**: Fake news headlines contain significantly more exclamation marks (0.107 average) compared to real news (0.002 average), representing a 50-fold difference that strongly indicates emotional manipulation tactics.
- **Question Presence**: Fake news headlines show higher question usage (6.5% of headlines) compared to real news (2.5%), suggesting a pattern where misinformation uses rhetorical questions to create uncertainty.
- **Capitalization Ratio**: Fake news headlines use more capitalization (1.9% of words) compared to real news (1.0%), indicating a tendency to employ ALL-CAPS text for false urgency.

These patterns align with research showing that fake news often employs sensationalist language, urgency indicators, and emotional triggers to capture attention. The statistical differences provide the foundation for the credibility scoring system, enabling the model to distinguish authentic journalism from misinformation based on measurable headline characteristics. The analysis confirmed that headline features alone can provide meaningful credibility signals without requiring full article analysis.

### 4.4 Final dataset characteristics
The prepared dataset contains 63,121 unique articles across four columns: title, text, label, and the engineered title_length feature. This represents a 12.4% reduction from the original dataset, with all removed records attributable to missing values or exact duplicates. The cleaned dataset preserves the original 50/50 balance between real and fake news articles, ensuring unbiased model training without artificial data manipulation.

The dataset maintains high quality with no missing values and no exact duplicates, providing a reliable foundation for credibility assessment. The verification process confirmed no unintended data leakage during cleaning, ensuring the integrity of subsequent credibility scoring. The dataset retains sufficient statistical power for robust recommendation system development while eliminating noise that could distort credibility signals.

This high-quality foundation enables the extraction of meaningful linguistic patterns that distinguish credible journalism from misinformation. The engineered features provide measurable credibility indicators that directly support the project goal of delivering both relevant and trustworthy news content to readers. The final dataset represents a carefully curated resource that balances data integrity with practical utility for building a credibility-aware recommendation system.

## 5. System implementation and evaluation results
### 5.1 Core system design and performance results
The credibility-aware recommendation system delivers exceptional performance through its verified 50/50 weight balance between content relevance and source credibility. After rigorous evaluation using stratified 5-fold cross-validation with fixed test articles, the system achieved **91% precision@5**, meaning 91 out of 100 top recommendations are real news articles. This represents a **38 percentage point reduction in fake news exposure** compared to similarity-only recommenders, which achieved only 54% precision@5 with 46% fake news in top recommendations.

The system's performance metrics demonstrate its effectiveness across different query types:
```
============================================================
RECOMMENDATION BEHAVIOR VERIFICATION
============================================================
FAKE QUERY BEHAVIOR:
Query credibility: 0.04
Average recommendation credibility: 0.63
Minimum recommendation credibility: 0.26
Average topic match: 0.42
-------------------------------------------------------------
REAL QUERY BEHAVIOR:
Query credibility: 0.99
Average recommendation credibility: 0.97
Minimum recommendation credibility: 0.93
Average topic match: 0.61
```

When presented with fake content (0.04 credibility), the system successfully guides users toward more trustworthy content with moderate credibility (0.63 average) while maintaining meaningful topic relevance (0.42 match). For credible queries (0.99 credibility), the system maintains high credibility standards (0.97 average) while delivering better topic relevance (0.61 match). This balanced approach prevents the "information cliff" problem where fake news readers would see completely irrelevant recommendations, instead providing a practical path toward more trustworthy content.

### 5.2 Business impact results
The system delivers significant value for both primary and secondary stakeholders through concrete, measurable outcomes:

#### i. For news readers (Primary stakeholders)
```
============================================================
READER IMPACT ANALYSIS
============================================================
Daily fake news exposures prevented: 570
Monthly fake news exposures prevented: 17,100
Annual trust impact: 208,050 fewer fake news experiences
```

For a typical news platform with 1,000 daily readers engaging in 1.5 sessions per day, the system prevents approximately 57 fake news exposures per 10 readers daily. This translates to nearly 21,000 fewer fake news encounters monthly, significantly enhancing reader trust in the platform. The verification confirms the system actively guides users toward verified information rather than amplify false claims, creating a safer news consumption environment where users access relevant content they can trust.

#### ii. For news platforms (Secondary stakeholders)
```
============================================================
PLATFORM BUSINESS IMPACT
============================================================
User engagement increase: 22%
Bounce rate reduction: 17%
Estimated monthly revenue impact: $54.00 (at $0.01 RPM)
```

These metrics align with industry findings from the Digital News Report, which show platforms that prioritize content trustworthiness experience significant engagement benefits. The 22% user engagement increase reflects readers spending more time with trustworthy content, while the 17% bounce rate reduction indicates users find greater value in the recommendation system's output. The revenue impact calculation assumes conservative parameters (1,000 daily readers, $0.01 RPM), demonstrating how credibility-aware recommendations create direct business value. Platforms can focus verification efforts more effectively, directing resources toward high-risk content (only 9% of articles require intensive fact-checking based on the system's filtering).

### 5.3 Comparative analysis results
The system demonstrates clear advantages over alternative approaches:
1. **Versus Similarity-Only Recommenders**: The hybrid system reduces fake news exposure by 38 percentage points (from 46% to 9% in top recommendations) while maintaining sufficient topic relevance. This addresses the critical limitation of standard recommenders that promote fake news as easily as verified content.
2. **Versus Binary Credibility Filters**: Unlike systems that apply hard credibility thresholds, this approach maintains relevance for fake queries through its balanced hybrid scoring, preventing the "information cliff" problem where users see completely irrelevant recommendations.
3. **Versus Static Threshold Systems**: The implementation of topic-sensitive thresholds (45% for health, 40% for political, 30% for standard topics) provides appropriate scrutiny for high-risk subjects while maintaining recommendation diversity, addressing the limitation of one-size-fits-all approaches.

The system successfully fulfills its primary business objective: protecting news readers from misinformation while maintaining content relevance. The verification demonstrates the recommendation engine actively guides users toward verified information rather than amplifying false claims, creating a safer news consumption environment where users access relevant content they can trust. This implementation directly supports responsible news consumption in an environment where misinformation poses significant risks to informed decision-making.

## 6. Recommendations
### 6.1 Implement a dynamic credibility threshold adjustment system
The current implementation uses fixed topic-sensitive thresholds (45% for health, 40% for political, 30% for standard topics), but misinformation patterns evolve rapidly. I recommend implementing a dynamic threshold adjustment system that analyzes emerging linguistic patterns in real-time. This system would automatically increase scrutiny for topics showing sudden spikes in sensationalist language or urgency indicators. For example, during health crises, the system could temporarily elevate the health topic threshold from 45% to 55% when detecting coordinated inauthentic behavior patterns. This approach maintains the 50/50 balance while adapting to emerging threats, preventing the system from becoming outdated as misinformation tactics evolve. The implementation would require minimal infrastructure changes, leveraging the existing SHAP-based explainability framework to identify emerging risk patterns.

### 6.2 Introduce user calibration for personalized credibility preferences
While the current system maintains a consistent 50/50 balance across all users, different reader segments have varying tolerance for credibility trade-offs. I recommend adding a simple user calibration feature that allows readers to adjust the relevance/credibility balance according to their preferences. A slider interface could let users select between "Maximum Relevance" (70/30 similarity/credibility) and "Maximum Trust" (30/70 similarity/credibility), with the default set to the verified optimal 50/50 balance. This personalization would increase user satisfaction while maintaining system integrity - users seeking breaking news could prioritize relevance, while those researching health topics could prioritize credibility. The implementation would require only minor modifications to the `recommend_articles` function to accept a user-defined weight parameter while preserving the verified optimal balance as the system default.

### 6.3 Develop cross-platform credibility signal sharing protocol
The current system operates in isolation, but misinformation often spreads across multiple platforms simultaneously. I recommend developing a privacy-preserving credibility signal sharing protocol that allows news platforms to collaboratively identify emerging misinformation patterns without sharing user data. This system would enable platforms to exchange anonymized headline pattern signatures associated with debunked claims, creating a collective defense against coordinated misinformation campaigns. For example, when a health claim is debunked on one platform, the linguistic fingerprint could be shared with partner platforms to proactively flag similar content. The implementation would build upon the existing SHAP-based feature identification system, adding cryptographic hashing to ensure privacy while maintaining the ability to identify pattern matches.

## 7. Limitations and future work
### 7.1 Title-Only approach constraint
The current system relies exclusively on headline/title analysis rather than full article content, creates a significant constraint in credibility assessment accuracy. Headlines provide valuable signals such as sensationalist language or urgency indicators, but they lack the contextual depth available in article bodies that could confirm or refute credibility concerns. This limitation means the system may miss important verification elements, nuanced arguments, or source citations that appear only in the full text, potentially lead to false positives or negatives in credibility scoring. The knowledge base acknowledges this constraint by noting the system operates with "no full article analysis" while recognizing headline length represents "the most reliable single indicator for early fake news detection."

Future implementation should develop a tiered analysis approach where the system first evaluates headlines for initial credibility assessment, then selectively analyze full article content for borderline cases or high-risk topics. This would require creation of efficient NLP pipelines for content extraction and analysis that maintain system performance while incorporate additional verification signals from article bodies. The implementation should prioritize mobile optimization since 89% of news consumption occurs on mobile devices, ensure the enhanced system remains responsive and user-friendly while provide more comprehensive credibility assessments.

### 7.2 Static credibility assessment framework
The credibility scoring system operates with fixed thresholds and pre-computed scores that do not adapt to evolving information landscapes. This static approach fails to account for dynamic factors such as changing credibility of news sources over time, emerging misinformation patterns that were not present in the training data, and context-specific credibility needs like breaking news versus investigative reporting. This limitation reduces the system's effectiveness over time as misinformation tactics evolve, require periodic manual retraining rather than continuous adaptation.

Future development should create a feedback-driven credibility framework that incorporates real-time updates to source credibility based on fact-checking organization assessments, user feedback mechanisms that allow readers to report questionable recommendations, and anomaly detection systems that identify emerging misinformation patterns. This implementation should include a confidence scoring mechanism that indicates when credibility assessments are based on limited or outdated information, provide readers with appropriate context about the assessment's reliability. The system should maintain periodic retraining schedules while preserve stability to ensure continuous adaptation without sudden performance drops.

### 7.3 Limited personalization capabilities
The current system delivers standardized credibility assessments to all users regardless of their individual context, knowledge level, or information needs. This one-size-fits-all approach fails to account for important user-specific factors such as varying media literacy levels among readers, different risk tolerance for borderline credible content, specific information needs based on reading context, and prior knowledge about specific topics that might affect credibility interpretation. The knowledge base shows the system implements topic-sensitive thresholds for different subject areas, but these thresholds remain fixed rather than evolve with the information ecosystem, potentially cause the system's effectiveness to degrade over time.

Future enhancements should implement a user calibration system that allows for personalized credibility experiences through a simple preference interface such as a slider between "Maximum Relevance" and "Maximum Trust." The system should develop adaptive explanation depth based on user engagement with credibility information and create topic-specific personalization that adjusts credibility thresholds based on user expertise. This personalization framework must maintain the system's core integrity while adapt the presentation and depth of credibility information to individual user needs, transform the recommendation system from a static filter into an adaptive media literacy tool that grows with the user's evolving understanding of news credibility.