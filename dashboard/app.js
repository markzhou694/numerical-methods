const modules = [
  {
    name: "Finite Differences",
    path: "finite_diff/",
    status: "Implemented",
    files: 4,
    summary:
      "Unified finite-difference builders use N intervals: fd_bvp_1d handles Dirichlet, Neumann, Robin, and mixed boundary data, while poisson_2d_matrix selects a 5- or 9-point stencil.",
    methods: ["1D linear BVP", "Robin boundary data", "Poisson stencil=5|9", "Richardson"],
    links: [
      ["fd_1d_bvp.py", "../finite_diff/fd_1d_bvp.py"],
      ["poisson_2d.py", "../finite_diff/poisson_2d.py"],
      ["richardson_extrap.py", "../finite_diff/richardson_extrap.py"],
    ],
  },
  {
    name: "ODE Solvers",
    path: "ode/",
    status: "Implemented",
    files: 4,
    summary:
      "Explicit Euler, implicit Backward Euler, Newton solves for nonlinear steps, adaptive RK23, and RK4 stability masks.",
    methods: ["Forward Euler", "Backward Euler", "Newton step", "RK23", "RK4 stability"],
    links: [
      ["forward_euler.py", "../ode/forward_euler.py"],
      ["backward_euler.py", "../ode/backward_euler.py"],
      ["rk23_adaptive.py", "../ode/rk23_adaptive.py"],
      ["rk4_stability.py", "../ode/rk4_stability.py"],
    ],
  },
  {
    name: "Integration",
    path: "integration/",
    status: "Implemented",
    files: 7,
    summary:
      "Composite rules, adaptive trapezoidal quadrature, Newton-Cotes weights, Gauss-Legendre quadrature, and tensor products.",
    methods: ["Trapezoidal", "Simpson", "Newton-Cotes", "Gauss-Legendre", "2D tensor product"],
    links: [
      ["trapezoidal.py", "../integration/trapezoidal.py"],
      ["simpson.py", "../integration/simpson.py"],
      ["adaptive_quad.py", "../integration/adaptive_quad.py"],
      ["newton_cotes.py", "../integration/newton_cotes.py"],
      ["gauss_legendre.py", "../integration/gauss_legendre.py"],
      ["tensor_product_2d.py", "../integration/tensor_product_2d.py"],
    ],
  },
  {
    name: "Interpolation",
    path: "interpolation/",
    status: "Implemented",
    files: 4,
    summary:
      "Lagrange interpolation on the standard interval with affine mapping and equispaced or Chebyshev nodes.",
    methods: ["Lagrange basis", "Chebyshev nodes", "Equispaced nodes", "Interval map"],
    links: [
      ["lagrange.py", "../interpolation/lagrange.py"],
      ["nodes.py", "../interpolation/nodes.py"],
      ["interval_map.py", "../interpolation/interval_map.py"],
    ],
  },
  {
    name: "Iterative Solvers",
    path: "iterative/",
    status: "Implemented",
    files: 9,
    summary:
      "Readable implementations of stationary methods, Newton iteration, projection methods, GMRES, and TriMRES.",
    methods: ["Jacobi", "Gauss-Seidel", "Conjugate Gradient", "Newton", "GMRES", "TriMRES", "Projection"],
    links: [
      ["jacobi.py", "../iterative/jacobi.py"],
      ["gauss_seidel.py", "../iterative/gauss_seidel.py"],
      ["conjugate_gradient.py", "../iterative/conjugate_gradient.py"],
      ["gmres.py", "../iterative/gmres.py"],
      ["trimres.py", "../iterative/trimres.py"],
      ["newton.py", "../iterative/newton.py"],
      ["greedy_projection.py", "../iterative/greedy_projection.py"],
      ["projection_2d.py", "../iterative/projection_2d.py"],
    ],
  },
  {
    name: "PDE and Utilities",
    path: "pde/, utils/",
    status: "Placeholder",
    files: 2,
    summary:
      "Package placeholders are present. The planned PDE examples are still listed as future work in the progress log.",
    methods: ["PDE placeholder", "Utility placeholder"],
    links: [
      ["pde/__init__.py", "../pde/__init__.py"],
      ["utils/__init__.py", "../utils/__init__.py"],
    ],
  },
];

const issues = [
  {
    priority: "P1",
    title: "Two-dimensional projection can hit a singular 2x2 system",
    file: "iterative/projection_2d.py",
    href: "../iterative/projection_2d.py",
    note:
      "When A r is parallel to r, the orthogonalized second direction is zero and the direct solve fails.",
  },
  {
    priority: "P2",
    title: "RK23 step-size update divides by zero for exact embedded agreement",
    file: "ode/rk23_adaptive.py",
    href: "../ode/rk23_adaptive.py",
    note:
      "A zero local error estimate currently proposes an infinite next step.",
  },
];

const roadmap = [
  {
    name: "Fourier",
    path: "fourier/",
    files: "dft_matrix.py, dft_matrix_odd.py, dft_matrix2.py, fourier_interpolant.py, algo3_pfpp.py, acyclic_conv.py",
    note: "A natural next area because it pairs well with interpolation and stability visualizations.",
  },
  {
    name: "Chebyshev",
    path: "chebyshev/",
    files: "cheb_diff_matrix.py, cheb_bvp.py, legendre_coeff_diff.py, legendre_pseudospectral_bvp.py",
    note: "Connects directly to the existing Chebyshev node work.",
  },
  {
    name: "Integral Equations",
    path: "integral_eq/",
    files: "nystrom_bie.py, fredholm_trapezoidal.py, fredholm_second_kind.py",
    note: "Good candidate after quadrature tests are in place.",
  },
  {
    name: "PDE",
    path: "pde/",
    files: "heat_adi.py, heat_nonlinear_be.py, schrodinger_explicit.py, schrodinger_cn_newton.py, advection_char.py",
    note: "Best added after ODE and finite-difference test coverage improves.",
  },
];

const moduleGrid = document.querySelector("#moduleGrid");
const issueList = document.querySelector("#issueList");
const roadmapList = document.querySelector("#roadmapList");
const searchInput = document.querySelector("#search");
const tabButtons = document.querySelectorAll(".tab-button");
const panels = document.querySelectorAll("[data-panel]");

function includesQuery(text, query) {
  return text.toLowerCase().includes(query.toLowerCase());
}

function renderModules(query = "") {
  moduleGrid.innerHTML = "";

  const filtered = modules.filter((module) => {
    const haystack = [
      module.name,
      module.path,
      module.status,
      module.summary,
      module.methods.join(" "),
      module.links.map(([label]) => label).join(" "),
    ].join(" ");
    return includesQuery(haystack, query);
  });

  if (filtered.length === 0) {
    moduleGrid.innerHTML = '<div class="empty-state">No matching modules or files.</div>';
    return;
  }

  for (const module of filtered) {
    const card = document.createElement("article");
    card.className = "module-card";
    card.innerHTML = `
      <div class="panel-heading">
        <h3>${module.name}</h3>
        <span class="badge ${module.status === "Implemented" ? "ready" : "planned"}">${module.status}</span>
      </div>
      <div class="module-meta">
        <span class="badge planned">${module.files} files</span>
        <span class="badge ready">${module.path}</span>
      </div>
      <p>${module.summary}</p>
      <p><strong>Methods:</strong> ${module.methods.join(", ")}</p>
      <div class="file-links">
        ${module.links.map(([label, href]) => `<a href="${href}">${label}</a>`).join("")}
      </div>
    `;
    moduleGrid.appendChild(card);
  }
}

function renderIssues() {
  issueList.innerHTML = "";
  for (const issue of issues) {
    const card = document.createElement("article");
    card.className = "issue-card";
    card.innerHTML = `
      <div class="issue-top">
        <strong>${issue.title}</strong>
        <span class="priority">${issue.priority}</span>
      </div>
      <p>${issue.note}</p>
      <a class="link-button" href="${issue.href}">${issue.file}</a>
    `;
    issueList.appendChild(card);
  }
}

function renderRoadmap() {
  roadmapList.innerHTML = "";
  for (const item of roadmap) {
    const card = document.createElement("article");
    card.className = "roadmap-item";
    card.innerHTML = `
      <div class="roadmap-top">
        <strong>${item.name}</strong>
        <span class="badge planned">${item.path}</span>
      </div>
      <p>${item.note}</p>
      <p><strong>Files:</strong> ${item.files}</p>
    `;
    roadmapList.appendChild(card);
  }
}

function setView(view) {
  for (const button of tabButtons) {
    const isActive = button.dataset.view === view;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  }

  for (const panel of panels) {
    panel.classList.toggle("hidden", panel.dataset.panel !== view);
  }
}

searchInput.addEventListener("input", (event) => {
  renderModules(event.target.value);
  setView("overview");
});

for (const button of tabButtons) {
  button.addEventListener("click", () => setView(button.dataset.view));
}

renderModules();
renderIssues();
renderRoadmap();
