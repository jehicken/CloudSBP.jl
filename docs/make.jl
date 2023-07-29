using CutDGD
using Documenter

DocMeta.setdocmeta!(CutDGD, :DocTestSetup, :(using CutDGD); recursive=true)

makedocs(;
    modules=[CutDGD],
    authors="Jason Hicken <jason.hicken@gmail.com> and contributors",
    repo="https://github.com/jehicken/CutDGD.jl/blob/{commit}{path}#{line}",
    sitename="CutDGD.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jehicken.github.io/CutDGD.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jehicken/CutDGD.jl",
    devbranch="main",
)
