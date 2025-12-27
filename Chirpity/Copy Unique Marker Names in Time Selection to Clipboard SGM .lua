-- @title        Copy Marker & Region Names by Frequency (Time Selection)
-- @author       Sruthin
-- @version      2.0.0
-- @about
--   Copies unique marker and region names that fall inside the current
--   time selection to the clipboard, sorted by how often they appear
--   (most frequent first). Output is comma-separated.
--
-- @requirements
--   SWS Extension (required for CF_SetClipboard)
--
-- @changelog
--   v1.0.0
--   - Initial release
--   - Frequency-based sorting of marker and region names
------------------------------------------------------------


-- Trim leading and trailing whitespace
function trim(s)
  return s:match("^%s*(.-)%s*$")
end

function main()
  local start_time, end_time =
    reaper.GetSet_LoopTimeRange(false, false, 0, 0, false)

  if end_time - start_time <= 0 then
    reaper.MB("Please create a time selection first.", "No Time Selection", 0)
    return
  end

  local _, num_markers, num_regions = reaper.CountProjectMarkers(0)
  local name_count = {}

  for i = 0, num_markers + num_regions - 1 do
    local retval, isrgn, pos, rgnend, name =
      reaper.EnumProjectMarkers(i)

    -- Markers inside time selection
    if not isrgn and pos >= start_time and pos <= end_time then
      name = trim(name)
      if name ~= "" then
        name_count[name] = (name_count[name] or 0) + 1
      end
    end

    -- Regions overlapping time selection
    if isrgn and rgnend > start_time and pos < end_time then
      name = trim(name)
      if name ~= "" then
        name_count[name] = (name_count[name] or 0) + 1
      end
    end
  end

  if not next(name_count) then
    reaper.MB(
      "No named markers or regions found in time selection.",
      "No Items",
      0
    )
    return
  end

  -- Sort by frequency (descending), then alphabetically
  local sorted_names = {}
  for name, count in pairs(name_count) do
    table.insert(sorted_names, { name = name, count = count })
  end

  table.sort(sorted_names, function(a, b)
    if a.count == b.count then
      return a.name:lower() < b.name:lower()
    end
    return a.count > b.count
  end)

  local result_names = {}
  for _, entry in ipairs(sorted_names) do
    table.insert(result_names, entry.name)
  end

  local final_text = table.concat(result_names, ", ")
  reaper.CF_SetClipboard(final_text) -- Requires SWS

  reaper.ShowMessageBox(
    "Copied marker/region names to clipboard (sorted by frequency):\n\n"
      .. final_text,
    "Success",
    0
  )
end

main()
