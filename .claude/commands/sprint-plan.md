---
description: Generate a comprehensive sprint plan based on existing Product Requirements Document (PRD) and Software Design Document (SDD)
allowed-tools: Task(sprint-planner:*), Read(docs/prd.md:*), Read(docs/sdd.md:*), Write(docs/sprint.md:*)
---

I'll help you create a comprehensive sprint plan based on your Product Requirements Document (PRD) and Software Design Document (SDD). Let me engage the sprint-planner agent to analyze your documents and generate an actionable development plan.

Use the Task tool with the sprint-planner agent to:

1. **Carefully read and analyze** both `docs/prd.md` and `docs/sdd.md` documents
2. **Ask clarifying questions** about any ambiguities in requirements, technical constraints, or business priorities, providing specific proposals for consideration
3. **Only proceed to generate the sprint plan** when all uncertainties are resolved and the agent has no remaining doubts
4. **Create a comprehensive sprint plan** with 2.5-day sprints covering:
   - Executive summary and timeline overview
   - Multiple sprint breakdowns (Sprint 1, Sprint 2, etc.)
   - Clear, measurable goals aligned with MVP objectives
   - Specific deliverables with detailed descriptions
   - Concrete acceptance criteria and progress checkboxes
   - Dependencies and risk mitigation strategies
   - Success metrics and review criteria

The agent will systematically clarify uncertainties about:
- Business priorities (must-have vs. nice-to-have features)
- Technical constraints or dependencies from the SDD
- Resource availability and team composition
- External dependencies or integrations
- MVP success criteria and definition of done

**Important**: The agent must resolve ALL uncertainties with you before proceeding. Only when completely satisfied with the answers and having no remaining doubts should it generate the final sprint plan and save it to `docs/sprint.md`.

Each sprint will deliver tangible, testable value while accounting for technical debt, testing, and integration time. The resulting plan will provide your development team with clear, actionable guidance for MVP delivery.