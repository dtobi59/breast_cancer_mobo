function Para(el)
  local text = pandoc.utils.stringify(el)
  text = text:gsub("([%%#&{}_])", "\\%1") -- escape LaTeX specials
  return pandoc.RawBlock("latex", "\\paragraph{" .. text .. "}")
end