import rl_mdp.util as util
from rl_mdp.model_free_prediction.monte_carlo_evaluator import MCEvaluator
from rl_mdp.model_free_prediction.td_evaluator import TDEvaluator
from rl_mdp.model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator

def main() -> None:
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """
    mdp = util.create_mdp()
    policy_1 = util.create_policy_1()
    policy_2 = util.create_policy_2()

    mc_evaluator = MCEvaluator(mdp)
    mc1 = mc_evaluator.evaluate(policy_1, 1000)
    mc2 = mc_evaluator.evaluate(policy_2, 1000)
    print("Policy 1 MC:", mc1)
    print("Policy 2 MC:", mc2)

    td_evaluator = TDEvaluator(mdp, 0.1)
    td1 = td_evaluator.evaluate(policy_1, 1000)
    td2 = td_evaluator.evaluate(policy_2, 1000)
    print("Policy 1 TD:", td1)
    print("Policy 2 TD:", td2)

    td_lambda_evaluator = TDEvaluator(mdp, 0.1)
    tdl1 = td_lambda_evaluator.evaluate(policy_1, 1000)
    tdl2 = td_lambda_evaluator.evaluate(policy_2, 1000)
    print("Policy 1 TD Lambda:", tdl1)
    print("Policy 2 TD Lambda:", tdl2)


if __name__ == "__main__":
    main()
