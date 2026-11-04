
import asyncio
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step, Event, Context, AgentWorkFlow
import random
from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_all_possible_flows_mermaid,
    draw_most_recent_execution,
    draw_most_recent_execution_mermaid,
)


class MyWorkFlow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="Hello World")

class ProcessingEvent(Event):
    intermediate_result: str

class LoopEvent(Event):
    loops: int

class MultiStepWorkFlow(Workflow):

    @step
    async def query(self, ctx: Context, ev: StartEvent) -> ProcessingEvent:
        await ctx.store.set("query","What is the capital of france")

    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        loops = 0
        if isinstance(ev, LoopEvent):
            loops = ev.loops
        if random.randint(0,1) == 0:
            return LoopEvent(loops=loops+1)
        else:
            return ProcessingEvent(intermediate_result=f"GooD, proceeed, retry {loops}")    

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result) 



async def main():
    w = MultiStepWorkFlow(timeout=10, verbose=False)

    workflow_context = Context(w)



    draw_all_possible_flows(w, "flow.html")

    # result = await w.run()
    # print(result)


if __name__ == "__main__":
    asyncio.run(main())
