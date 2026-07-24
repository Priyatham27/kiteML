"""
test_dag.py — Unit tests for TrainingDAG and TrainingNode (Story 5.1 Flagship Feature).
"""

from kiteml.training import TrainingContext, TrainingDAG, TrainingNode


class MockNode(TrainingNode):

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.executed = False

    def prepare(self, context: TrainingContext) -> None:
        pass

    def execute(self, context: TrainingContext) -> None:
        self.executed = True


def test_training_dag_execution():
    dag = TrainingDAG()
    node = MockNode("TestNode")
    dag.add_node(node)

    ctx = TrainingContext()
    dag.execute_all(ctx)

    assert node.executed is True
