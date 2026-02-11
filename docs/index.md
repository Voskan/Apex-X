# Apex-X Documentation

Welcome to the **Apex-X** documentation portal. Apex-X is a dynamic compute graph runtime for efficient computer vision on the edge.

---

<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting Started**
    ---
    Quickly set up your environment and run your first model.

    [:octicons-arrow-right-24: Installation](../README.md#quickstart-cpu)
    [:octicons-arrow-right-24: GPU Setup](../README.md#quickstart-gpu)

-   :material-chart-bar: **Benchmarks**
    ---
    See how Apex-X compares to YOLOv8, YOLO26, and RT-DETR.

    [:octicons-arrow-right-24: Performance Report](benchmarks.md)

-   :material-cogs: **Engineering Spec**
    ---
    Deep dive into the architecture, math, and implementation details.

    [:octicons-arrow-right-24: Read the Spec](ENGINEERING_SPEC.md)
    [:octicons-arrow-right-24: Dual Routing Math](algorithms.md)

-   :material-api: **Runtime & Plugins**
    ---
    Learn about TensorRT plugins and deployment contracts.

    [:octicons-arrow-right-24: Plugin Spec](runtime/PLUGIN_SPEC.md)
    [:octicons-arrow-right-24: TensorRT Integration](runtime/TENSORRT.md)

</div>

## 🏗 System Architecture

The core of Apex-X is its **Dual-Stream** architecture, which separates low-resolution context processing from high-resolution detail extraction.

```mermaid
graph TD
    subgraph Stream_Input ["Input Stage"]
        I[Input Image] --> Pre[Pre-Processing]
    end

    subgraph Stream_PV ["Peripheral Vision (Coarse)"]
        Pre --> PV_Backbone[PV Backbone]
        PV_Backbone --> PV_Feat["Dense 1/16 Features"]
    end

    subgraph Router_Logic ["Dynamic Routing"]
        PV_Feat --> Router{Utility Network}
        Router -->|Compute Budget| Budget[Budget Controller]
        Budget -->|Select Top K Tiles| Mask[Active Tile Mask]
        Mask -->|K <= Kmax| Packer[Tile Packer]
    end

    subgraph Stream_FF ["Foveal Focus (High-Res)"]
        Pre --> Packer
        Packer -->|"Packed Batch [B, K, C, t, t]"| FF_Backbone[FF Backbone]
        FF_Backbone -->|Refined Features| Rec["Tile-SSM / Refinement"]
        Rec --> Unpacker[Tile Unpacker]
    end

    subgraph Stream_Fusion ["Fusion & Output"]
        PV_Feat --> Merger[Feature Fusion]
        Unpacker --> Merger
        Merger --> Head[Detection Head]
        Head --> Output["Bounding Boxes / Masks"]
    end

    style Router fill:#f55,stroke:#333
    style Budget fill:#fa5,stroke:#333
    style Packer fill:#55f,stroke:#333
    style Unpacker fill:#55f,stroke:#333
```

## 📚 Topics

### Core Concepts

1.  **[Algorithms](algorithms.md)**: The mathematical foundation of dual-variable routing and continuous budgeting.
2.  **[Context](CONTEXT.md)**: The "Project Memory" containing architectural decisions and history.
3.  **[PRD](PRD.md)**: The Product Requirements Document defining the scope of Apex-X v4.

### Release Notes

-   **[v4 Release Checklist](release/CHECKLIST.md)**: Status of the current release candidate.
-   **[Migration Guide](release/MIGRATION.md)**: How to upgrade from previous versions.

### Developer Guides

-   **[Contributing](../CONTRIBUTING.md)**: How to contribute code and docs.
-   **[Decisions Log](DECISIONS.md)**: Record of architectural decision records (ADRs).
