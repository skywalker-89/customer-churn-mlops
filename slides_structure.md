# Slide Deck Structure: Toy Store Optimization
**Presenter**: Lead Engineer & Lead Data Scientist
**Goal**: Justify Data Selection & Problem Setting (30pts)

## Section 1: The Problem (The "Why")
**Slide 1: Title Slide**
*   **Title**: "Smart Funnels: Optimizing Toy Store Revenue with MLOps"
*   **Subtitle**: From Raw Clicks to Real Revenue.

**Slide 2: The Business Challenge**
*   **Context**: Maven Fuzzy Factory (Online Toy Store).
*   **The Problem**: "Marketing brings people to the door, but 94% leave without buying."
*   **The Solution**: An end-to-end ML System to:
    1.  **Predict Conversion** (Who will buy?).
    2.  **Estimate Value** (How much?).

## Section 2: Infrastructure (Lead Engineer Role)
*Focus: "Building the Factory"*

**Slide 3: The Architecture (Show Diagram)**
*   **Visual**: Draw a simple flow: `CSV -> Airflow -> MinIO -> Feature Engine -> Parquet`.
*   **Key Points**:
    *   **Airflow**: Orchestrates the ingestion of 6+ relational tables.
    *   **MinIO**: "Data Lake" storage for scalability (ELT pattern).
    *   **Parquet**: Optimized columnar storage for ML training.

**Slide 4: Proof of Implementation**
*   **Visual**: **SCREENSHOT of your Airflow UI** (showing green bars).
*   **Visual**: **SCREENSHOT of your MinIO Bucket** (showing `website_sessions.parquet`, etc.).
*   **Narrative**: "We didn't just analyze a CSV in Excel. We built a production-grade ingestion pipeline."

## Section 3: Data Strategy (Lead Data Scientist Role)
*Focus: "Designing the Intelligence"*

**Slide 5: Feature Engineering Strategy (The "5+ Features")**
*   **Visual**: A Table listing your features (Copy from Handoff Report).
    *   `Traffic Source` (Google vs Social)
    *   `Device Type` (Mobile vs Desktop)
    *   `Time of Day` (Behavioral)
    *   `Weekend` (Temporal)
    *   `Repeat User` (History)
*   **Narrative**: "We engineered features that capture *User Intent*, not just User Demographics."

**Slide 6: The Target Variables**
*   **Visual**: Show the code snippet where you created `is_ordered` and `revenue`.
*   **Explain**:
    *   **Classification**: `is_ordered` (0/1). Challenge: High Class Imbalance (6.8%).
    *   **Regression**: `revenue` (USD). Challenge: Long-tail distribution.

**Slide 7: Data Evidence**
*   **Visual**: **SCREENSHOT of your Terminal Output** from `feature_engineering.py`.
    *   "Rows: 472,871"
    *   "Columns: [hour_of_day, is_weekend, ...]"
*   **Narrative**: "We have processed 472k user sessions and are ready to hand off clean, One-Hot Encoded data to the ML Engineers."
