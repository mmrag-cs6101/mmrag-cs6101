---
description: Generate a comprehensive Software Design Document (SDD) from an existing Product Requirements Document (PRD)
allowed-tools: Task(ai-product-architect:*), Read(docs/prd.md:*), Write(docs/sdd.md:*)
---

I'll help you create a comprehensive Software Design Document (SDD) based on your Product Requirements Document (PRD). Let me engage the ai-product-architect agent to carefully analyze your PRD and generate a detailed technical architecture.

Use the Task tool with the ai-product-architect agent to:

1. **Carefully read and analyze** the PRD at `docs/prd.md`
2. **Ask clarifying questions** about any technical ambiguities or missing requirements, providing specific proposals for consideration
3. **Only proceed to generate the SDD** when all uncertainties are resolved and the agent has no remaining doubts
4. **Create a comprehensive SDD** covering:
   - Project Architecture (system design, components, data flow)
   - Software Stack (languages, frameworks, tools, deployment)
   - AI Models and Integration (ML pipeline, inference, monitoring)
   - Data Architecture (databases, pipelines, governance)
   - UI/UX Architecture (frontend design, state management)

The agent will systematically work through technical clarifications on:
- Performance and scalability requirements
- Security and compliance considerations
- Integration with existing systems
- Budget/resource constraints
- Timeline impact on architecture decisions

**Important**: The agent must clarify ALL uncertainties with you before proceeding. Only when completely satisfied with the answers and having no remaining doubts should it generate the final SDD and save it to `docs/sdd.md`.

The resulting SDD will serve as a comprehensive technical blueprint ready for engineering teams to begin sprint planning and implementation.