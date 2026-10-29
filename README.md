# Governed Prompt Gateway
**Token-Efficient and Safety-Aware LLM Routing for Regulated Customers**

**Program:** Break Through Tech AI Studio - Fall 2026  
**Company / Org:** Other  
**Challenge Advisor:** Bhavana Unnam (unnam29@gmail.com)  
**AI Studio Coach:** Om Kamath (om.kamath@breakthroughtech.org)  

---

## 👥 Team Members

| Name | GitHub Handle | Contribution |
| :--- | :--- | :--- |
| David Evers | @davidevers | Data exploration, visualization, overall project coordination |
| Aaron Li | @aazurezt | Data preprocessing, feature engineering, data validation |
| Ruth Assefa | TBD | Model selection, hyperparameter tuning, model training |
| Sangam Subedi | TBD | Model evaluation, performance analysis, results interpretation |
| Abeih Hamani | TBD | Caching implementation, token optimization |
| Tung Nguyen | TBD | Governance layer, risk classification testing |

*(Note: Update GitHub handles and specific contributions as the project progresses).*

---

## 🎯 Project Highlights
- **Developed an AI governance framework** utilizing NLP classification, semantic similarity, prompt optimization, and RAG to secure generative AI models in high-stakes industries.
- **Optimized operational costs** by reducing LLM token usage and estimating cost savings across simulated customer-service workloads.
- **Implemented a risk-aware routing gateway** that screens user queries, blocks harmful requests, and recommends the best LLM model tier prior to execution.
- **Created an auditable decision pipeline** to meet the rigorous compliance standards required by regulated customer service environments.

---

## 🏗️ Project Overview
This project addresses the critical business challenge of deploying generative AI safely and cost-effectively in regulated environments. We built a governed prompt gateway designed to evaluate and route user queries before they reach an LLM. 

By leveraging publicly available datasets (customer-service, insurance Q&A, toxic prompts, jailbreaks, and hallucinations), the gateway actively screens for safety, optimizes safe prompts to reduce token count, and utilizes semantic caching to improve latency. This multi-layered approach reduces unnecessary LLM spend, improves response reliability, and creates a transparent governance layer for enterprise customer service use cases.

---

## 📊 Success Criteria & Targets
1. **Safety and Governance Accuracy:**
   - Achieve a strong baseline performance on harmful-query and risk classification.
   - Target: **80%+ F1 score** on prompt-risk classification (measuring accuracy, precision, and recall).
2. **Token Reduction and Cost Efficiency:**
   - Compress and optimize inputs compared to original prompt token counts.
   - Target: **30–50% reduction** in input tokens for eligible prompts, estimating cost savings across simulated workloads.
3. **Cache Effectiveness:**
   - Measure and maximize cache-hit rates across test sets containing repeated and semantically similar queries.

---

## 🚀 The Gateway Demo Workflow
Our final pipeline demonstrates the following end-to-end execution:
1. **Query Input:** User enters a prompt.
2. **Classification:** System evaluates query intent and risk.
3. **Governance:** Unsafe or toxic queries are explicitly blocked or redirected.
4. **Optimization:** Safe queries are compressed for token reduction.
5. **Caching:** Semantically similar queries trigger a cache reuse to save compute.
6. **Dashboard & Auditing:** The UI displays token savings, cost estimates, risk scores, and the recommended model tier with an auditable decision log.

---

## 🗺️ Project Milestones

| Month | Milestone | Key Activities |
| :--- | :--- | :--- |
| **September** | Problem Definition & Baseline Design | Finalize scope and criteria. Select datasets and conduct EDA. Define risk categories and governance rules. Build data pipeline and baseline query classifier for intent/risk. Establish baseline metrics. |
| **October** | Governance, Optimization & Caching | Build initial prompt governance layer. Implement harmful-query and jailbreak detection. Add prompt compression/refinement. Implement semantic caching. Measure token reduction. |
| **November** | Dashboard & Final Demo | Add complexity/cost-based model recommendation logic. Finalize the dashboard (token savings, cache status, governance decisions). Evaluate system accuracy and hit rates. Prepare final architecture and demo. |

---

## 🌟 Stretch Goals & Next Steps
- **Jurisdiction-Aware Policy Profiles:** Create configurable policy templates for different regulated sectors (insurance, banking, healthcare) to demonstrate how enterprise AI governance rules can be customized regionally.
- **Automated Prompt Testing Suite:** Build a test harness to run bulk prompts through the gateway, automatically generating reports on safety accuracy, token reduction, cache performance, and routing consistency.

---

## 👩🏽‍💻 Setup and Installation
*(Provide step-by-step instructions so someone else can run your code and reproduce your results)*
- How to clone the repository (`git clone ...`)
- How to install dependencies (`pip install -r requirements.txt`)
- How to set up the environment and access the datasets
- How to run the notebook, scripts, or React frontend

---

## 📝 License
This project is licensed under the [MIT License](LICENSE).
