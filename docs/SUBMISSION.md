# Hackathon entry

**GridPFN — local forecasting, federated home-energy control, inspectable decisions**

## Submission description

GridPFN helps a household ask when to charge, cool, store solar energy or run
appliances—and inspect the bill and comfort tradeoffs. TabPFN-3.5 uses the home's
labeled history to forecast six hours of demand, solar and temperature. These
forecasts feed a small controller trained collaboratively with FedAvg. A local
assistant and MCP tools expose forecasts, schedule comparisons and explanations;
an optional LLM routes requests to numerical tools.

The evaluation compares TabPFN with TabFM, TabICLv2 and three controls across
25 homes and five forward monthly folds. Policies are refitted on all eligible
training and validation dates before their test month. Independent simulator
replay and a certified perfect-future oracle make physical consistency and the
remaining objective gap explicit. The repository includes the process diagram,
complete evidence explorer and train-to-home-model workflow.

## Reviewer path

1. Read the README and process diagram.
2. Open the hosted interactive comparison, or run `python demo.py` locally.
3. Follow the assistant's generated-data demo without restricted household files.
4. Inspect the protocol, complete method/month results and reproduction commands.

This entry fits **Build an agent**, **Build an extension or app**, and **Take on a
hard problem**. TabPFN supplies predictions; PPO supplies numerical control; the
optional LLM provides a natural-language interface through structured tools.
Simulation evidence is distinct from physical deployment or guaranteed savings.

## Submission links

- Repository: [Nousphera/GridPFN](https://github.com/Nousphera/GridPFN)
- Website: [GridPFN](https://nousphera.github.io/GridPFN/)
- Interactive comparison: [foundation models and oracle](https://nousphera.github.io/GridPFN/performance.html?scope=foundations)

Code is licensed under Apache-2.0. Model weights and household inputs retain
separate terms. The repository contains aggregate results and a generated-data
demo; private inputs and trained weights are not redistributed.
The hackathon form has not been submitted.

Source: [official hackathon page and terms](https://platform.priorlabs.ai/hackathon-3.5).
