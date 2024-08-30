using CloudSBP
using Documenter

DocMeta.setdocmeta!(CloudSBP, :DocTestSetup, :(using CloudSBP); recursive=true)

makedocs(;
    modules=[CloudSBP],
    authors="Jason Hicken <jason.hicken@gmail.com> and contributors",
    repo="https://github.com/jehicken/CloudSBP.jl/blob/{commit}{path}#{line}",
    sitename="CloudSBP.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jehicken.github.io/CloudSBP.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jehicken/CloudSBP.jl",
    devbranch="main",
)
