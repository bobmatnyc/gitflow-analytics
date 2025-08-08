# ChatGPT Integration Setup

GitFlow Analytics now supports ChatGPT-4.1 for generating qualitative executive summaries in the narrative report.

## Setup

1. **Get an OpenAI API Key**
   - Sign up at [platform.openai.com](https://platform.openai.com)
   - Navigate to API Keys section
   - Create a new API key

2. **Set the Environment Variable**
   
   Add to your `.env` file (in the same directory as your config.yaml):
   ```bash
   OPENAI_API_KEY=sk-...your-api-key-here...
   ```
   
   Or export it in your shell:
   ```bash
   export OPENAI_API_KEY="sk-...your-api-key-here..."
   ```

3. **Run Analysis**
   
   When you run the analysis with markdown output enabled, ChatGPT will automatically generate a qualitative executive summary:
   ```bash
   gitflow-analytics analyze -c config.yaml --weeks 4
   ```

## What ChatGPT Analyzes

The ChatGPT integration provides:

- **Executive Summary**: 2-3 paragraphs highlighting overall team performance
- **Strategic Insights**: Deep analysis of team dynamics and project health
- **Actionable Recommendations**: 3-5 specific recommendations for leadership
- **Risk Assessment**: Identification of potential issues and mitigation strategies

## Output

The ChatGPT analysis appears in the narrative report under a new "Qualitative Analysis" section after the standard executive summary metrics.

## Cost Considerations

- Each analysis uses approximately 1,000-1,500 tokens
- With GPT-4-turbo pricing, this costs approximately $0.01-0.02 per analysis
- The integration only runs when OPENAI_API_KEY is set

## Fallback Behavior

If ChatGPT is unavailable or fails:
- A basic algorithmic summary is generated instead
- The analysis continues without interruption
- A warning message indicates the fallback was used

## Example Output

```markdown
## Qualitative Analysis

Over the past 4 weeks, the development team has demonstrated exceptional momentum with a 
significant increase in velocity, delivering 280 commits across 13 active projects. The 
team health score of 75/100 reflects a well-functioning unit, though there are opportunities 
for improvement in cross-team collaboration.

The concentration of work among top contributors (Luca-Borda with 30.4% of commits) suggests 
strong technical leadership but also indicates potential knowledge silos. The 34.3% ticket 
coverage reveals a gap in process adherence that could impact traceability and project 
management effectiveness.

### Strategic Insights

1. **Productivity Patterns**: The team shows consistent afternoon productivity peaks, 
   suggesting potential for morning standups or collaborative sessions
2. **Project Distribution**: FRONTEND_PROJECT dominates with 77.3% of activity, 
   indicating either strategic focus or resource imbalance
3. **Developer Specialization**: High focus scores (100% for several developers) suggest 
   deep expertise but limited knowledge sharing

### Recommendations

1. **Implement Pair Programming**: Address knowledge concentration by rotating developers 
   across projects
2. **Enhance Ticket Discipline**: Target 60%+ ticket coverage through pre-commit hooks 
   or workflow automation
3. **Balance Workload**: Consider redistributing tasks from high-volume contributors to 
   prevent burnout
4. **Morning Collaboration**: Leverage natural afternoon productivity by scheduling 
   collaborative work in mornings
5. **Cross-Project Reviews**: Institute weekly cross-project code reviews to break down silos
```