1. Press record in UI
2. Humans speak for a bit
3. Press stop recording
4. UI posts text from recording to backend
5. Backend sends text to openai
6. Pass OpenAi generated promopt tp perplexity
7. Write response from perplexity to txt file locally
8. Push txt file to Knowledge Base 
9. Human launches conversation with agent via UI
10. Human asks agent what its got
11. Agent talk but also uses tool to open relelvant URL