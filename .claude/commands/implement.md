---
description: Implement development tasks from sprint plan with comprehensive review cycle
allowed-tools: Task(sprint-implementer:*), Read(docs/sprint.md:*), Read(docs/prd.md:*), Read(docs/sdd.md:*), Read(docs/a2a/engineer-feedback.md:*), Write(docs/a2a/reviewer.md:*), Write(*:*), Edit(*:*), Bash(*:*)
argument-hint: "sprint-number or task description"
---

I'll help you implement the specified sprint tasks using the sprint-implementer agent. This command follows a comprehensive development and review cycle to ensure high-quality deliverables.

Use the Task tool with the sprint-implementer agent to implement: $ARGUMENTS

**Implementation Process:**

1. **Pre-Implementation Review** (if applicable):
   - Check for `docs/a2a/engineer-feedback.md` from previous reviews
   - Address any feedback before starting new implementation
   - Clarify uncertainties with the reviewer if needed

2. **Context Analysis**:
   - Review `docs/sprint.md` for assigned tasks
   - Study `docs/prd.md` for product requirements and context
   - Examine `docs/sdd.md` for system design and architecture
   - Analyze existing codebase for patterns and standards

3. **Full Implementation**:
   - Implement ALL assigned development tasks in the sprint
   - Follow existing code patterns, architecture, and standards
   - Write clean, maintainable, well-documented code
   - Ensure proper error handling and edge case coverage
   - Maintain consistency with project structure and conventions

4. **Comprehensive Testing**:
   - Create unit tests for all implemented functionality
   - Follow existing testing patterns and frameworks
   - Ensure high test coverage with meaningful test cases
   - Include positive, negative, and edge case scenarios

5. **Implementation Report**:
   - Generate detailed report at `docs/a2a/reviewer.md`
   - Include: completed tasks, implementation approach, testing coverage
   - Document challenges encountered and recommendations
   - Provide comprehensive details for senior technical product lead review

**Review Cycle:**
- Senior technical product lead reviews `docs/a2a/reviewer.md`
- If issues found, feedback provided in `docs/a2a/engineer-feedback.md`
- Agent reads feedback, clarifies if needed, and fixes issues
- Updated report generated at `docs/a2a/reviewer.md` for re-review
- Cycle continues until sprint completion meets quality standards

This ensures thorough implementation with proper quality gates and stakeholder oversight.