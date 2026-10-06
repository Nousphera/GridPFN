"""The same bounded home-energy tools for an external MCP-connected LLM."""

import contextlib
import sys

from mcp.server.mcpserver import MCPServer

from .periods import PeriodService
from .presentation import tool_response
from .service import EvidenceService


def serve(directory, live=False, home=None):
    service = EvidenceService(directory, live=live)
    service.bind_home(home)
    mcp = MCPServer("GridPFN Home")

    @mcp.tool()
    def list_homes() -> dict:
        """Find the homes and dates you can explore. Start here if no home is selected."""
        catalog = service.catalog()
        return {
            "homes": catalog["homes"],
            "data_source": "Generated example homes"
            if catalog["provenance"]["kind"] == "synthetic"
            else "Household research data",
            "next_step": "Choose a home and date, then ask about costs, expected electricity use or a different energy plan.",
        }

    @mcp.tool()
    def inspect_home(
        home: str, date: str, include_details: bool = False, end_date: str | None = None
    ) -> dict:
        """Learn about this home's energy history, devices and available plans.

        Use a home and date from list_homes. Set end_date for a week, month or custom period; missing days are reported. Ask for details only when the user
        wants the model setup or supporting data.
        """
        return tool_response(
            PeriodService(service, date, end_date), "inspect_home", home, date, include_details
        )

    @mcp.tool()
    def explain_bill(
        home: str, date: str, include_details: bool = False, end_date: str | None = None
    ) -> dict:
        """Answer 'Why is my electricity bill high?' with costs and expensive hours.

        Set end_date to total a week, month or custom period; gaps are reported.
        Uses this home's readings and prices, not a utility invoice. It cannot
        diagnose faulty equipment. include_details adds the hourly calculations.
        """
        return tool_response(
            PeriodService(service, date, end_date), "explain_bill", home, date, include_details
        )

    @mcp.tool()
    def forecast_and_explain(
        home: str, date: str, include_details: bool = False, end_date: str | None = None
    ) -> dict:
        """Explain how much electricity TabPFN expects and what shaped its estimate.

        Set end_date to average the daily forecasts for a period, not to forecast an entire future month.
        Factors explain the forecast, not the causes of a bill. The forecast
        covers everyday use, excluding appliances whose schedules we can change.
        include_details adds the method, forecast series and model information.
        """
        with contextlib.redirect_stdout(sys.stderr):
            return tool_response(
                PeriodService(service, date, end_date),
                "forecast_and_explain",
                home,
                date,
                include_details,
            )

    @mcp.tool()
    def compare_schedules(
        home: str, date: str, include_details: bool = False, end_date: str | None = None
    ) -> dict:
        """Answer 'Could a different energy plan cost less?' and show the trade-offs.

        Set end_date to compare the same plans across a period. Days are tested separately.
        Compares plans tested on the selected past days. Keep simulated savings
        clearly labeled; never promise the same savings on a future bill.
        Plans must meet charging, laundry, comfort and battery checks to qualify.
        include_details adds hourly schedules and calculation assumptions.
        """
        return tool_response(
            PeriodService(service, date, end_date), "compare_schedules", home, date, include_details
        )

    @mcp.tool()
    def export_plan(home: str, date: str, end_date: str | None = None) -> dict:
        """Get a complete home energy report to review or save.

        Includes costs, forecasts, explanations and the plans tested. Nothing in
        the home is changed. The report includes technical details for checking.
        """
        with contextlib.redirect_stdout(sys.stderr):
            return tool_response(PeriodService(service, date, end_date), "export_plan", home, date)

    @mcp.tool()
    def appliance_breakdown(home: str, date: str, end_date: str | None = None) -> dict:
        """Show which appliances cost most, with solar credits and fixed charges.

        Based on recorded energy traces; gross appliance costs are not avoidable savings.
        Set end_date for any available calendar period.
        """
        return tool_response(
            PeriodService(service, date, end_date), "appliance_breakdown", home, date
        )

    @mcp.tool()
    def schedule_comparison(
        home: str, date: str, end_date: str | None = None, plan: str | None = None
    ) -> dict:
        """Compare the selected plan with a cheaper tested alternative, hour by hour.

        Both are simulations priced with observed tariffs, not a record of what the
        person did or a globally optimal schedule. Every daily constraint is checked.
        """
        return tool_response(
            PeriodService(service, date, end_date, plan), "schedule_comparison", home, date
        )

    @mcp.tool()
    def plan_day(
        home: str, date: str, end_date: str | None = None, plan: str | None = None, hour: int = 6
    ) -> dict:
        """Show when cooling, car charging and laundry fit a forecast-based day plan.

        Pick hour 0, 6, 12 or 18. Uses forecasts and a dated simulated state, not live
        readings. Prices are a persistence assumption plus the known tariff. No actuation.
        """
        return tool_response(
            PeriodService(service, date, end_date, plan, hour), "plan_day", home, date
        )

    @mcp.tool()
    def explain_forecast_inputs(
        home: str, date: str, origin: int = 6, target: int = 9, variable: str = "demand"
    ) -> dict:
        """Explain a demand, solar or temperature estimate with recent readings and SHAP.

        Choose an origin and a future target within the trained horizon (six hours
        for study models). Includes units and exact grouped Shapley contributions.
        These explain a forecast, not causes, policy actions or guaranteed savings.
        A period should be inspected one date at a time. Requires live TabPFN.
        """
        service.select(home, date)
        if service.live is None:
            return {
                "available": False,
                "note": "Enable local TabPFN inference to inspect forecast inputs.",
            }
        with contextlib.redirect_stdout(sys.stderr):
            return service.live(
                home, date, origin_hour=origin, target_hour=target, target_name=variable
            )

    mcp.run(transport="stdio")
