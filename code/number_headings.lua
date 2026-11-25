-- Prefix decimal numbers to headings (levels 1-3); headings with class "unnumbered" are skipped.
local c = {0, 0, 0}
function Header(h)
  if h.classes:includes('unnumbered') or h.level > 3 then return h end
  c[h.level] = c[h.level] + 1
  for i = h.level + 1, 3 do c[i] = 0 end
  local n = tostring(c[1])
  for i = 2, h.level do n = n .. '.' .. c[i] end
  table.insert(h.content, 1, pandoc.Space())
  table.insert(h.content, 1, pandoc.Str(n))
  return h
end
