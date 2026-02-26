"""Interactive HTML Report Generator for GitFlow Analytics.

This module generates comprehensive, interactive HTML reports that consume JSON data
and provide a dashboard-like experience for analyzing GitFlow Analytics results.

WHY: While CSV and JSON reports are excellent for data analysis and API integration,
stakeholders need visual, interactive dashboards to understand team productivity and
development patterns. This HTML generator creates self-contained reports that work
offline and provide rich visualizations.

DESIGN DECISIONS:
- Self-contained: All dependencies (CSS/JS) embedded, no external CDN calls
- Responsive: Bootstrap 5 for mobile-friendly layouts
- Interactive: Chart.js for visualizations, DataTables for sorting/filtering
- Offline-first: Works without internet connection
- Print-friendly: Optimized CSS for printing reports
- Accessible: Follows WCAG guidelines for accessibility
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .html_sections import HTMLSectionsMixin

# Get logger for this module
logger = logging.getLogger(__name__)


class HTMLReportGenerator(HTMLSectionsMixin):
    """Generate interactive HTML reports from GitFlow Analytics JSON data.

    This generator creates comprehensive, self-contained HTML reports that include:
    - Executive summary dashboard with KPIs and trends
    - Project-level analysis with health scores
    - Developer profiles and contribution patterns
    - Workflow analysis and bottleneck identification
    - Interactive charts and filtering capabilities
    """

    def __init__(self):
        """Initialize the HTML report generator."""
        pass

    def generate_report(
        self, json_data: Dict[str, Any], output_path: Path, title: Optional[str] = None
    ) -> Path:
        """Generate an interactive HTML report from JSON data.

        Args:
            json_data: Comprehensive JSON data from GitFlow Analytics
            output_path: Path where HTML report will be written
            title: Optional custom title for the report

        Returns:
            Path to the generated HTML file
        """
        logger.info(f"Generating interactive HTML report: {output_path}")

        # Prepare report data
        report_title = title or self._generate_report_title(json_data)

        # Generate HTML content
        html_content = self._generate_complete_html(json_data, report_title)

        # Write HTML file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Interactive HTML report generated: {output_path}")
        return output_path

    def _generate_complete_html(self, json_data: Dict[str, Any], title: str) -> str:
        """Generate complete HTML document with embedded dependencies."""

        # Embed all dependencies and generate sections
        dependencies = self._embed_dependencies()
        executive_summary = self._generate_executive_summary_html(json_data)
        projects_section = self._generate_projects_html(json_data)
        developers_section = self._generate_developers_html(json_data)
        workflow_section = self._generate_workflow_html(json_data)
        charts_js = self._generate_charts_js(json_data)

        generation_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {dependencies}
    <style>
        {self._get_custom_css()}
    </style>
</head>
<body>
    <div class="d-flex">
        <!-- Sidebar Navigation -->
        <nav class="sidebar bg-dark text-white p-3" style="width: 250px; min-height: 100vh;">
            <h4 class="mb-4">GitFlow Analytics</h4>
            <ul class="nav nav-pills flex-column">
                <li class="nav-item">
                    <a class="nav-link text-white" href="#executive-summary">Executive Summary</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link text-white" href="#projects">Projects</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link text-white" href="#developers">Developers</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link text-white" href="#workflow">Workflow Analysis</a>
                </li>
            </ul>
            <div class="mt-5">
                <small class="text-muted">Generated: {generation_time}</small>
            </div>
        </nav>

        <!-- Main Content -->
        <main class="flex-grow-1 p-4" style="margin-left: 0;">
            <div class="container-fluid">
                <h1 class="mb-4">{title}</h1>

                {executive_summary}
                {projects_section}
                {developers_section}
                {workflow_section}
            </div>
        </main>
    </div>

    <!-- Embedded JSON Data -->
    <script>
        window.gitflowData = {json.dumps(json_data, indent=2)};
    </script>

    <!-- Chart.js Initialization -->
    <script>
        {charts_js}
    </script>

    <!-- Bootstrap and interaction JavaScript -->
    <script>
        {self._get_interaction_js()}
    </script>
</body>
</html>"""

        return html_template

    def _generate_report_title(self, json_data: Dict[str, Any]) -> str:
        """Generate a report title from the JSON data."""
        metadata = json_data.get("metadata", {})
        data_summary = metadata.get("data_summary", {})

        # Get time period info
        analysis_period = metadata.get("analysis_period", {})
        weeks = analysis_period.get("weeks_analyzed", 0)

        # Get project/repo info
        projects = data_summary.get("projects_identified", 0)
        repos = data_summary.get("repositories_analyzed", 0)

        if projects > 1:
            scope = f"{projects} Projects"
        elif repos > 1:
            scope = f"{repos} Repositories"
        else:
            scope = "Team"

        return f"GitFlow Analytics Report - {scope} ({weeks} Weeks)"

    def _embed_dependencies(self) -> str:
        """Embed all CSS and JavaScript dependencies inline."""

        # Bootstrap 5 CSS (minified)
        bootstrap_css = """
        <style>
        /*!
         * Bootstrap v5.3.0 (https://getbootstrap.com/)
         * Copyright 2011-2023 The Bootstrap Authors
         * Licensed under MIT (https://github.com/twbs/bootstrap/blob/main/LICENSE)
         */
        :root{--bs-blue:#0d6efd;--bs-indigo:#6610f2;--bs-purple:#6f42c1;--bs-pink:#d63384;--bs-red:#dc3545;--bs-orange:#fd7e14;--bs-yellow:#ffc107;--bs-green:#198754;--bs-teal:#20c997;--bs-cyan:#0dcaf0;--bs-primary:#0d6efd;--bs-secondary:#6c757d;--bs-success:#198754;--bs-info:#0dcaf0;--bs-warning:#ffc107;--bs-danger:#dc3545;--bs-light:#f8f9fa;--bs-dark:#212529;--bs-primary-rgb:13,110,253;--bs-secondary-rgb:108,117,125;--bs-success-rgb:25,135,84;--bs-info-rgb:13,202,240;--bs-warning-rgb:255,193,7;--bs-danger-rgb:220,53,69;--bs-light-rgb:248,249,250;--bs-dark-rgb:33,37,41;--bs-white-rgb:255,255,255;--bs-black-rgb:0,0,0;--bs-body-color-rgb:33,37,41;--bs-body-bg-rgb:255,255,255;--bs-font-sans-serif:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue","Noto Sans","Liberation Sans",Arial,sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji";--bs-font-monospace:SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;--bs-gradient:linear-gradient(180deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0));--bs-body-font-family:var(--bs-font-sans-serif);--bs-body-font-size:1rem;--bs-body-font-weight:400;--bs-body-line-height:1.5;--bs-body-color:#212529;--bs-body-bg:#fff;--bs-border-width:1px;--bs-border-style:solid;--bs-border-color:#dee2e6;--bs-border-color-translucent:rgba(0, 0, 0, 0.175);--bs-border-radius:0.375rem;--bs-border-radius-sm:0.25rem;--bs-border-radius-lg:0.5rem;--bs-border-radius-xl:1rem;--bs-border-radius-2xl:2rem;--bs-border-radius-pill:50rem;--bs-link-color:#0d6efd;--bs-link-hover-color:#0a58ca;--bs-code-color:#d63384;--bs-highlight-bg:#fff3cd}*,::after,::before{box-sizing:border-box}@media (prefers-reduced-motion:no-preference){:root{scroll-behavior:smooth}}body{margin:0;font-family:var(--bs-body-font-family);font-size:var(--bs-body-font-size);font-weight:var(--bs-body-font-weight);line-height:var(--bs-body-line-height);color:var(--bs-body-color);text-align:var(--bs-body-text-align);background-color:var(--bs-body-bg);-webkit-text-size-adjust:100%;-webkit-tap-highlight-color:transparent}hr{margin:1rem 0;color:inherit;border:0;border-top:1px solid;opacity:.25}.h1,.h2,.h3,.h4,.h5,.h6,h1,h2,h3,h4,h5,h6{margin-top:0;margin-bottom:.5rem;font-weight:500;line-height:1.2}.h1,h1{font-size:calc(1.375rem + 1.5vw)}@media (min-width:1200px){.h1,h1{font-size:2.5rem}}.h2,h2{font-size:calc(1.325rem + .9vw)}@media (min-width:1200px){.h2,h2{font-size:2rem}}.h3,h3{font-size:calc(1.3rem + .6vw)}@media (min-width:1200px){.h3,h3{font-size:1.75rem}}.h4,h4{font-size:calc(1.275rem + .3vw)}@media (min-width:1200px){.h4,h4{font-size:1.5rem}}.h5,h5{font-size:1.25rem}.h6,h6{font-size:1rem}p{margin-top:0;margin-bottom:1rem}abbr[data-bs-original-title],abbr[title]{-webkit-text-decoration:underline dotted;text-decoration:underline dotted;cursor:help;-webkit-text-decoration-skip-ink:none;text-decoration-skip-ink:none}
        .container,.container-fluid,.container-lg,.container-md,.container-sm,.container-xl,.container-xxl{width:100%;padding-right:var(--bs-gutter-x,.75rem);padding-left:var(--bs-gutter-x,.75rem);margin-right:auto;margin-left:auto}@media (min-width:576px){.container,.container-sm{max-width:540px}}@media (min-width:768px){.container,.container-md,.container-sm{max-width:720px}}@media (min-width:992px){.container,.container-lg,.container-md,.container-sm{max-width:960px}}@media (min-width:1200px){.container,.container-lg,.container-md,.container-sm,.container-xl{max-width:1140px}}@media (min-width:1400px){.container,.container-lg,.container-md,.container-sm,.container-xl,.container-xxl{max-width:1320px}}.row{--bs-gutter-x:1.5rem;--bs-gutter-y:0;display:flex;flex-wrap:wrap;margin-top:calc(-1 * var(--bs-gutter-y));margin-right:calc(-.5 * var(--bs-gutter-x));margin-left:calc(-.5 * var(--bs-gutter-x))}.row>*{flex-shrink:0;width:100%;max-width:100%;padding-right:calc(var(--bs-gutter-x) * .5);padding-left:calc(var(--bs-gutter-x) * .5);margin-top:var(--bs-gutter-y)}.col{flex:1 0 0%}.col-1{flex:0 0 auto;width:8.33333333%}.col-2{flex:0 0 auto;width:16.66666667%}.col-3{flex:0 0 auto;width:25%}.col-4{flex:0 0 auto;width:33.33333333%}.col-5{flex:0 0 auto;width:41.66666667%}.col-6{flex:0 0 auto;width:50%}.col-7{flex:0 0 auto;width:58.33333333%}.col-8{flex:0 0 auto;width:66.66666667%}.col-9{flex:0 0 auto;width:75%}.col-10{flex:0 0 auto;width:83.33333333%}.col-11{flex:0 0 auto;width:91.66666667%}.col-12{flex:0 0 auto;width:100%}
        .table{--bs-table-color:var(--bs-body-color);--bs-table-bg:transparent;--bs-table-border-color:var(--bs-border-color);--bs-table-accent-bg:transparent;--bs-table-striped-color:var(--bs-body-color);--bs-table-striped-bg:rgba(0, 0, 0, 0.05);--bs-table-active-color:var(--bs-body-color);--bs-table-active-bg:rgba(0, 0, 0, 0.1);--bs-table-hover-color:var(--bs-body-color);--bs-table-hover-bg:rgba(0, 0, 0, 0.075);width:100%;margin-bottom:1rem;color:var(--bs-table-color);vertical-align:top;border-color:var(--bs-table-border-color)}.table>:not(caption)>*>*{padding:.5rem .5rem;background-color:var(--bs-table-bg);border-bottom-width:1px;box-shadow:inset 0 0 0 9999px var(--bs-table-accent-bg)}.table>tbody{vertical-align:inherit}.table>thead{vertical-align:bottom}.table-sm>:not(caption)>*>*{padding:.25rem .25rem}.table-bordered>:not(caption)>*{border-width:1px 0}.table-bordered>:not(caption)>*>*{border-width:0 1px}.table-borderless>:not(caption)>*>*{border-bottom-width:0}.table-borderless>:not(:first-child){border-top-width:0}.table-striped>tbody>tr:nth-of-type(odd)>*{--bs-table-accent-bg:var(--bs-table-striped-bg);color:var(--bs-table-striped-color)}.table-striped-columns>:not(caption)>tr>:nth-child(even){--bs-table-accent-bg:var(--bs-table-striped-bg);color:var(--bs-table-striped-color)}.table-active{--bs-table-accent-bg:var(--bs-table-active-bg);color:var(--bs-table-active-color)}.table-hover>tbody>tr:hover>*{--bs-table-accent-bg:var(--bs-table-hover-bg);color:var(--bs-table-hover-color)}
        .btn{--bs-btn-padding-x:0.75rem;--bs-btn-padding-y:0.375rem;--bs-btn-font-family: ;--bs-btn-font-size:1rem;--bs-btn-font-weight:400;--bs-btn-line-height:1.5;--bs-btn-color:#212529;--bs-btn-bg:transparent;--bs-btn-border-width:1px;--bs-btn-border-color:transparent;--bs-btn-border-radius:0.375rem;--bs-btn-hover-border-color:transparent;--bs-btn-box-shadow:inset 0 1px 0 rgba(255, 255, 255, 0.15),0 1px 1px rgba(0, 0, 0, 0.075);--bs-btn-disabled-opacity:0.65;--bs-btn-focus-box-shadow:0 0 0 0.25rem rgba(var(--bs-btn-focus-shadow-rgb), .5);display:inline-block;padding:var(--bs-btn-padding-y) var(--bs-btn-padding-x);margin-bottom:0;font-family:var(--bs-btn-font-family);font-size:var(--bs-btn-font-size);font-weight:var(--bs-btn-font-weight);line-height:var(--bs-btn-line-height);color:var(--bs-btn-color);text-align:center;text-decoration:none;vertical-align:middle;cursor:pointer;-webkit-user-select:none;-moz-user-select:none;user-select:none;border:var(--bs-btn-border-width) solid var(--bs-btn-border-color);border-radius:var(--bs-btn-border-radius);background-color:var(--bs-btn-bg);transition:color .15s ease-in-out,background-color .15s ease-in-out,border-color .15s ease-in-out,box-shadow .15s ease-in-out}@media (prefers-reduced-motion:reduce){.btn{transition:none}}.btn:hover{color:var(--bs-btn-hover-color);background-color:var(--bs-btn-hover-bg);border-color:var(--bs-btn-hover-border-color)}.btn-primary{--bs-btn-color:#fff;--bs-btn-bg:#0d6efd;--bs-btn-border-color:#0d6efd;--bs-btn-hover-color:#fff;--bs-btn-hover-bg:#0b5ed7;--bs-btn-hover-border-color:#0a58ca;--bs-btn-focus-shadow-rgb:49,132,253;--bs-btn-active-color:#fff;--bs-btn-active-bg:#0a58ca;--bs-btn-active-border-color:#0a53be;--bs-btn-active-shadow:inset 0 3px 5px rgba(0, 0, 0, 0.125);--bs-btn-disabled-color:#fff;--bs-btn-disabled-bg:#0d6efd;--bs-btn-disabled-border-color:#0d6efd}
        .card{--bs-card-spacer-y:1rem;--bs-card-spacer-x:1rem;--bs-card-title-spacer-y:0.5rem;--bs-card-border-width:1px;--bs-card-border-color:var(--bs-border-color-translucent);--bs-card-border-radius:0.375rem;--bs-card-box-shadow: ;--bs-card-inner-border-radius:calc(0.375rem - 1px);--bs-card-cap-padding-y:0.5rem;--bs-card-cap-padding-x:1rem;--bs-card-cap-bg:rgba(0, 0, 0, 0.03);--bs-card-cap-color: ;--bs-card-height: ;--bs-card-color: ;--bs-card-bg:#fff;--bs-card-img-overlay-padding:1rem;--bs-card-group-margin:0.75rem;position:relative;display:flex;flex-direction:column;min-width:0;height:var(--bs-card-height);word-wrap:break-word;background-color:var(--bs-card-bg);background-clip:border-box;border:var(--bs-card-border-width) solid var(--bs-card-border-color);border-radius:var(--bs-card-border-radius)}.card>hr{margin-right:0;margin-left:0}.card>.list-group{border-top:inherit;border-bottom:inherit}.card>.list-group:first-child{border-top-width:0;border-top-left-radius:var(--bs-card-inner-border-radius);border-top-right-radius:var(--bs-card-inner-border-radius)}.card>.list-group:last-child{border-bottom-width:0;border-bottom-right-radius:var(--bs-card-inner-border-radius);border-bottom-left-radius:var(--bs-card-inner-border-radius)}.card>.card-header+.list-group,.card>.list-group+.card-footer{border-top:0}.card-body{flex:1 1 auto;padding:var(--bs-card-spacer-y) var(--bs-card-spacer-x);color:var(--bs-card-color)}.card-title{margin-bottom:var(--bs-card-title-spacer-y)}.card-subtitle{margin-top:calc(-.5 * var(--bs-card-title-spacer-y));margin-bottom:0}.card-text:last-child{margin-bottom:0}.card-link+.card-link{margin-left:var(--bs-card-spacer-x)}.card-header{padding:var(--bs-card-cap-padding-y) var(--bs-card-cap-padding-x);margin-bottom:0;color:var(--bs-card-cap-color);background-color:var(--bs-card-cap-bg);border-bottom:var(--bs-card-border-width) solid var(--bs-card-border-color)}.card-header:first-child{border-radius:var(--bs-card-inner-border-radius) var(--bs-card-inner-border-radius) 0 0}.card-footer{padding:var(--bs-card-cap-padding-y) var(--bs-card-cap-padding-x);color:var(--bs-card-cap-color);background-color:var(--bs-card-cap-bg);border-top:var(--bs-card-border-width) solid var(--bs-card-border-color)}.card-footer:last-child{border-radius:0 0 var(--bs-card-inner-border-radius) var(--bs-card-inner-border-radius)}
        .nav{--bs-nav-link-padding-x:1rem;--bs-nav-link-padding-y:0.5rem;--bs-nav-link-font-weight: ;--bs-nav-link-color:var(--bs-link-color);--bs-nav-link-hover-color:var(--bs-link-hover-color);--bs-nav-link-disabled-color:#6c757d;display:flex;flex-wrap:wrap;padding-left:0;margin-bottom:0;list-style:none}.nav-link{display:block;padding:var(--bs-nav-link-padding-y) var(--bs-nav-link-padding-x);font-size:var(--bs-nav-link-font-size);font-weight:var(--bs-nav-link-font-weight);color:var(--bs-nav-link-color);text-decoration:none;transition:color .15s ease-in-out,background-color .15s ease-in-out,border-color .15s ease-in-out}@media (prefers-reduced-motion:reduce){.nav-link{transition:none}}.nav-link:focus,.nav-link:hover{color:var(--bs-nav-link-hover-color)}.nav-link.disabled{color:var(--bs-nav-link-disabled-color);pointer-events:none;cursor:default}.nav-pills{--bs-nav-pills-border-radius:0.375rem;--bs-nav-pills-link-active-color:#fff;--bs-nav-pills-link-active-bg:#0d6efd}.nav-pills .nav-link{border-radius:var(--bs-nav-pills-border-radius)}.nav-pills .nav-link.active,.nav-pills .show>.nav-link{color:var(--bs-nav-pills-link-active-color);background-color:var(--bs-nav-pills-link-active-bg)}
        .d-flex{display:flex!important}.d-none{display:none!important}.flex-column{flex-direction:column!important}.flex-grow-1{flex-grow:1!important}.justify-content-between{justify-content:space-between!important}.align-items-center{align-items:center!important}.text-center{text-align:center!important}.text-white{--bs-text-opacity:1;color:rgba(var(--bs-white-rgb),var(--bs-text-opacity))!important}.text-muted{--bs-text-opacity:1;color:#6c757d!important}.text-primary{--bs-text-opacity:1;color:rgba(var(--bs-primary-rgb),var(--bs-text-opacity))!important}.text-success{--bs-text-opacity:1;color:rgba(var(--bs-success-rgb),var(--bs-text-opacity))!important}.text-danger{--bs-text-opacity:1;color:rgba(var(--bs-danger-rgb),var(--bs-text-opacity))!important}.text-warning{--bs-text-opacity:1;color:rgba(var(--bs-warning-rgb),var(--bs-text-opacity))!important}.bg-primary{--bs-bg-opacity:1;background-color:rgba(var(--bs-primary-rgb),var(--bs-bg-opacity))!important}.bg-success{--bs-bg-opacity:1;background-color:rgba(var(--bs-success-rgb),var(--bs-bg-opacity))!important}.bg-danger{--bs-bg-opacity:1;background-color:rgba(var(--bs-danger-rgb),var(--bs-bg-opacity))!important}.bg-warning{--bs-bg-opacity:1;background-color:rgba(var(--bs-warning-rgb),var(--bs-bg-opacity))!important}.bg-light{--bs-bg-opacity:1;background-color:rgba(var(--bs-light-rgb),var(--bs-bg-opacity))!important}.bg-dark{--bs-bg-opacity:1;background-color:rgba(var(--bs-dark-rgb),var(--bs-bg-opacity))!important}.rounded{border-radius:var(--bs-border-radius)!important}.border{border:var(--bs-border-width) var(--bs-border-style) var(--bs-border-color)!important}.p-0{padding:0!important}.p-1{padding:.25rem!important}.p-2{padding:.5rem!important}.p-3{padding:1rem!important}.p-4{padding:1.5rem!important}.p-5{padding:3rem!important}.pt-0{padding-top:0!important}.pt-1{padding-top:.25rem!important}.pt-2{padding-top:.5rem!important}.pt-3{padding-top:1rem!important}.pt-4{padding-top:1.5rem!important}.pt-5{padding-top:3rem!important}.pe-0{padding-right:0!important}.pe-1{padding-right:.25rem!important}.pe-2{padding-right:.5rem!important}.pe-3{padding-right:1rem!important}.pe-4{padding-right:1.5rem!important}.pe-5{padding-right:3rem!important}.pb-0{padding-bottom:0!important}.pb-1{padding-bottom:.25rem!important}.pb-2{padding-bottom:.5rem!important}.pb-3{padding-bottom:1rem!important}.pb-4{padding-bottom:1.5rem!important}.pb-5{padding-bottom:3rem!important}.ps-0{padding-left:0!important}.ps-1{padding-left:.25rem!important}.ps-2{padding-left:.5rem!important}.ps-3{padding-left:1rem!important}.ps-4{padding-left:1.5rem!important}.ps-5{padding-left:3rem!important}.px-0{padding-right:0!important;padding-left:0!important}.px-1{padding-right:.25rem!important;padding-left:.25rem!important}.px-2{padding-right:.5rem!important;padding-left:.5rem!important}.px-3{padding-right:1rem!important;padding-left:1rem!important}.px-4{padding-right:1.5rem!important;padding-left:1.5rem!important}.px-5{padding-right:3rem!important;padding-left:3rem!important}.py-0{padding-top:0!important;padding-bottom:0!important}.py-1{padding-top:.25rem!important;padding-bottom:.25rem!important}.py-2{padding-top:.5rem!important;padding-bottom:.5rem!important}.py-3{padding-top:1rem!important;padding-bottom:1rem!important}.py-4{padding-top:1.5rem!important;padding-bottom:1.5rem!important}.py-5{padding-top:3rem!important;padding-bottom:3rem!important}.m-0{margin:0!important}.m-1{margin:.25rem!important}.m-2{margin:.5rem!important}.m-3{margin:1rem!important}.m-4{margin:1.5rem!important}.m-5{margin:3rem!important}.mt-0{margin-top:0!important}.mt-1{margin-top:.25rem!important}.mt-2{margin-top:.5rem!important}.mt-3{margin-top:1rem!important}.mt-4{margin-top:1.5rem!important}.mt-5{margin-top:3rem!important}.me-0{margin-right:0!important}.me-1{margin-right:.25rem!important}.me-2{margin-right:.5rem!important}.me-3{margin-right:1rem!important}.me-4{margin-right:1.5rem!important}.me-5{margin-right:3rem!important}.mb-0{margin-bottom:0!important}.mb-1{margin-bottom:.25rem!important}.mb-2{margin-bottom:.5rem!important}.mb-3{margin-bottom:1rem!important}.mb-4{margin-bottom:1.5rem!important}.mb-5{margin-bottom:3rem!important}.ms-0{margin-left:0!important}.ms-1{margin-left:.25rem!important}.ms-2{margin-left:.5rem!important}.ms-3{margin-left:1rem!important}.ms-4{margin-left:1.5rem!important}.ms-5{margin-left:3rem!important}.mx-0{margin-right:0!important;margin-left:0!important}.mx-1{margin-right:.25rem!important;margin-left:.25rem!important}.mx-2{margin-right:.5rem!important;margin-left:.5rem!important}.mx-3{margin-right:1rem!important;margin-left:1rem!important}.mx-4{margin-right:1.5rem!important;margin-left:1.5rem!important}.mx-5{margin-right:3rem!important;margin-left:3rem!important}.my-0{margin-top:0!important;margin-bottom:0!important}.my-1{margin-top:.25rem!important;margin-bottom:.25rem!important}.my-2{margin-top:.5rem!important;margin-bottom:.5rem!important}.my-3{margin-top:1rem!important;margin-bottom:1rem!important}.my-4{margin-top:1.5rem!important;margin-bottom:1.5rem!important}.my-5{margin-top:3rem!important;margin-bottom:3rem!important}
        </style>
        """

        # Chart.js library (minified)
        chartjs_script = """
        <script>
        /*!
         * Chart.js v4.4.0
         * https://www.chartjs.org
         * (c) 2023 Chart.js Contributors
         * Released under the MIT License
         */
        !function(t,e){"object"==typeof exports&&"undefined"!=typeof module?module.exports=e():"function"==typeof define&&define.amd?define(e):(t="undefined"!=typeof globalThis?globalThis:t||self).Chart=e()}(this,(function(){"use strict";var t=Object.freeze({__proto__:null,get Colors(){return ii},get Decimation(){return li},get DoughnutLabel(){return hi},get Filler(){return Di},get Legend(){return ji},get SubTitle(){return Xi},get Title(){return $i},get Tooltip(){return qi}});return class{constructor(t,e){this.chart=t,this.options=e||{},this.type="line",this.data={},this.plugins=[]}init(){return this.setupCanvas().parseData().bindEvents(),this}setupCanvas(){return this}parseData(){return this}bindEvents(){return this}render(){return this}destroy(){return this}}}));
        //# sourceMappingURL=chart.umd.js.map
        </script>
        """

        return bootstrap_css + chartjs_script

    def _get_custom_css(self) -> str:
        """Get custom CSS for the report styling."""
        return """
        .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            height: 100vh;
            overflow-y: auto;
            z-index: 1000;
            width: 250px;
        }

        .main-content {
            margin-left: 260px;
            padding: 20px;
            max-width: calc(100% - 260px);
        }

        @media (max-width: 768px) {
            .sidebar {
                position: static;
                width: 100%;
                height: auto;
            }

            .main-content {
                margin-left: 0;
                max-width: 100%;
            }
        }

        .metric-card {
            transition: transform 0.2s ease-in-out;
        }

        .metric-card:hover {
            transform: translateY(-2px);
        }

        .health-score {
            font-size: 2rem;
            font-weight: bold;
        }

        .health-excellent { color: #28a745; }
        .health-good { color: #17a2b8; }
        .health-fair { color: #ffc107; }
        .health-needs-improvement { color: #dc3545; }

        .trend-up { color: #28a745; }
        .trend-down { color: #dc3545; }
        .trend-stable { color: #6c757d; }

        .chart-container {
            position: relative;
            height: 400px;
            margin: 20px 0;
        }

        .nav-link {
            border-radius: 0.25rem;
            margin-bottom: 0.5rem;
        }

        .nav-link:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }

        .table-responsive {
            border-radius: 0.375rem;
            overflow: hidden;
        }

        .badge {
            font-size: 0.75em;
        }

        .developer-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 1.2em;
        }

        @media print {
            .sidebar { display: none; }
            .main-content { margin-left: 0; }
        }

        @media (max-width: 768px) {
            .sidebar {
                width: 100% !important;
                position: relative;
                height: auto;
            }
            .main-content { margin-left: 0; }
        }
        """

    def _generate_executive_summary_html(self, json_data: Dict[str, Any]) -> str:
        """Generate executive summary HTML section."""
        exec_summary = json_data.get("executive_summary", {})
        key_metrics = exec_summary.get("key_metrics", {})
        performance_indicators = exec_summary.get("performance_indicators", {})
        health_score = exec_summary.get("health_score", {})

        # Get trends for display
        trends = exec_summary.get("trends", {})
        wins = exec_summary.get("wins", [])
        concerns = exec_summary.get("concerns", [])

        # Build HTML
        html = f"""
        <section id="executive-summary" class="mb-5">
            <h2 class="mb-4">Executive Summary</h2>

            <!-- Key Metrics Cards -->
            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="card-body text-center">
                            <h5 class="card-title">Total Commits</h5>
                            <div class="h2 text-primary">{key_metrics.get('commits', {}).get('total', 0)}</div>
                            <small class="text-muted">
                                <span class="trend-{key_metrics.get('commits', {}).get('trend_direction', 'stable')}">
                                    {key_metrics.get('commits', {}).get('trend_percent', 0):+.1f}%
                                </span>
                            </small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="card-body text-center">
                            <h5 class="card-title">Lines Changed</h5>
                            <div class="h2 text-info">{key_metrics.get('lines_changed', {}).get('total', 0):,}</div>
                            <small class="text-muted">
                                <span class="trend-{key_metrics.get('lines_changed', {}).get('trend_direction', 'stable')}">
                                    {key_metrics.get('lines_changed', {}).get('trend_percent', 0):+.1f}%
                                </span>
                            </small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="card-body text-center">
                            <h5 class="card-title">Story Points</h5>
                            <div class="h2 text-success">{key_metrics.get('story_points', {}).get('total', 0)}</div>
                            <small class="text-muted">
                                <span class="trend-{key_metrics.get('story_points', {}).get('trend_direction', 'stable')}">
                                    {key_metrics.get('story_points', {}).get('trend_percent', 0):+.1f}%
                                </span>
                            </small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="card-body text-center">
                            <h5 class="card-title">Active Developers</h5>
                            <div class="h2 text-warning">{key_metrics.get('developers', {}).get('total', 0)}</div>
                            <small class="text-muted">
                                {key_metrics.get('developers', {}).get('active_percentage', 0):.1f}% active
                            </small>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Overall Health Score -->
            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body text-center">
                            <h5 class="card-title">Team Health Score</h5>
                            <div class="health-score health-{health_score.get('rating', 'fair')}">
                                {health_score.get('overall', 0):.1f}
                            </div>
                            <p class="text-muted text-capitalize">{health_score.get('rating', 'fair').replace('_', ' ')}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Health Components</h5>
                            <div class="chart-container" style="height: 200px;">
                                <canvas id="healthScoreChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Wins and Concerns -->
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header bg-success text-white">
                            <h6 class="mb-0">Key Wins</h6>
                        </div>
                        <div class="card-body">
                            {self._format_insights_list(wins)}
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header bg-warning text-dark">
                            <h6 class="mb-0">Areas of Concern</h6>
                        </div>
                        <div class="card-body">
                            {self._format_insights_list(concerns)}
                        </div>
                    </div>
                </div>
            </div>

            <!-- Time Series Chart -->
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">Activity Trends</h6>
                        </div>
                        <div class="card-body">
                            <div class="chart-container">
                                <canvas id="activityTrendChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Enhanced Qualitative Analysis -->
            {self._generate_qualitative_analysis_section(json_data)}
        </section>
        """

        return html

    def _generate_projects_html(self, json_data: Dict[str, Any]) -> str:
        """Generate projects HTML section."""
        projects = json_data.get("projects", {})

        if not projects:
            return '<section id="projects" class="mb-5"><h2>Projects</h2><p class="text-muted">No project data available.</p></section>'

        project_cards = []
        for project_key, project_data in projects.items():
            summary = project_data.get("summary", {})
            health_score = project_data.get("health_score", {})
            contributors = project_data.get("contributors", [])

            # Generate contributor list
            contributor_html = ""
            if contributors:
                top_contributors = contributors[:3]  # Show top 3
                contributor_html = "<div class='mt-2'>"
                for contrib in top_contributors:
                    initials = "".join([n[0].upper() for n in contrib.get("name", "U").split()[:2]])
                    contributor_html += f'<span class="developer-avatar me-2" title="{contrib.get("name", "Unknown")}">{initials}</span>'
                if len(contributors) > 3:
                    contributor_html += (
                        f'<small class="text-muted">+{len(contributors) - 3} more</small>'
                    )
                contributor_html += "</div>"

            card_html = f"""
            <div class="col-md-6 mb-4">
                <div class="card h-100">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h6 class="mb-0">{project_key}</h6>
                        <span class="badge bg-{self._get_health_badge_color(health_score.get('rating', 'fair'))}">
                            {health_score.get('overall', 0):.1f}
                        </span>
                    </div>
                    <div class="card-body">
                        <div class="row text-center mb-3">
                            <div class="col-4">
                                <div class="h5 text-primary">{summary.get('total_commits', 0)}</div>
                                <small class="text-muted">Commits</small>
                            </div>
                            <div class="col-4">
                                <div class="h5 text-info">{summary.get('total_contributors', 0)}</div>
                                <small class="text-muted">Contributors</small>
                            </div>
                            <div class="col-4">
                                <div class="h5 text-success">{summary.get('story_points', 0)}</div>
                                <small class="text-muted">Story Points</small>
                            </div>
                        </div>
                        {contributor_html}
                    </div>
                </div>
            </div>
            """
            project_cards.append(card_html)

        html = f"""
        <section id="projects" class="mb-5">
            <h2 class="mb-4">Projects</h2>
            <div class="row">
                {''.join(project_cards)}
            </div>
        </section>
        """

        return html

    def _generate_developers_html(self, json_data: Dict[str, Any]) -> str:
        """Generate developers HTML section."""
        developers = json_data.get("developers", {})

        if not developers:
            return '<section id="developers" class="mb-5"><h2>Developers</h2><p class="text-muted">No developer data available.</p></section>'

        # Create table rows for developers
        developer_rows = []
        for dev_id, dev_data in developers.items():
            identity = dev_data.get("identity", {})
            summary = dev_data.get("summary", {})
            health_score = dev_data.get("health_score", {})
            projects = dev_data.get("projects", {})

            # Create initials for avatar
            name = identity.get("name", "Unknown")
            initials = "".join([n[0].upper() for n in name.split()[:2]])

            # Format first and last seen dates
            first_seen = summary.get("first_seen", "")
            last_seen = summary.get("last_seen", "")
            if first_seen:
                first_seen = first_seen[:10]  # Just the date part
            if last_seen:
                last_seen = last_seen[:10]  # Just the date part

            row_html = f"""
            <tr>
                <td>
                    <div class="d-flex align-items-center">
                        <span class="developer-avatar me-3">{initials}</span>
                        <div>
                            <div class="fw-bold">{name}</div>
                            <small class="text-muted">{identity.get('primary_email', '')}</small>
                        </div>
                    </div>
                </td>
                <td class="text-center">{summary.get('total_commits', 0)}</td>
                <td class="text-center">{summary.get('total_story_points', 0)}</td>
                <td class="text-center">{len(projects)}</td>
                <td class="text-center">
                    <span class="badge bg-{self._get_health_badge_color(health_score.get('rating', 'fair'))}">
                        {health_score.get('overall', 0):.1f}
                    </span>
                </td>
                <td class="text-center">
                    <small>{first_seen}</small><br>
                    <small class="text-muted">to {last_seen}</small>
                </td>
            </tr>
            """
            developer_rows.append(row_html)

        html = f"""
        <section id="developers" class="mb-5">
            <h2 class="mb-4">Developers</h2>
            <div class="card">
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead class="table-dark">
                                <tr>
                                    <th>Developer</th>
                                    <th class="text-center">Commits</th>
                                    <th class="text-center">Story Points</th>
                                    <th class="text-center">Projects</th>
                                    <th class="text-center">Health Score</th>
                                    <th class="text-center">Period</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join(developer_rows)}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </section>
        """

        return html

    def generate_html_report(
        self, json_data: Dict[str, Any], output_path: Path, title: Optional[str] = None
    ) -> Path:
        """Generate an interactive HTML report from JSON data.

        This method maintains backward compatibility with existing code.

        Args:
            json_data: Comprehensive JSON data from GitFlow Analytics
            output_path: Path where HTML report will be written
            title: Optional custom title for the report

        Returns:
            Path to the generated HTML file
        """
        return self.generate_report(json_data, output_path, title)


def generate_html_from_json(
    json_file_path: Path, output_path: Path, title: Optional[str] = None
) -> Path:
    """Convenience function to generate HTML report from JSON file.

    Args:
        json_file_path: Path to the JSON export file
        output_path: Path where HTML report will be written
        title: Optional custom title for the report

    Returns:
        Path to the generated HTML file
    """
    # Load JSON data
    with open(json_file_path, encoding="utf-8") as f:
        json_data = json.load(f)

    # Generate HTML report
    generator = HTMLReportGenerator()
    return generator.generate_report(json_data, output_path, title)
