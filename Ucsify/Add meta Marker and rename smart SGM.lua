-- @description Rename takes from clipboard Filename and append META markers
-- @version 2.1.0
-- @author SGM
-- @about
--   # Rename Takes from Clipboard + META Markers
--
--   This script reads a UCS-style `Filename="..."` string from the clipboard
--   and renames selected REAPER takes accordingly.
--
--   ## Behaviour
--   - Keeps the plain filename if no duplicates exist
--   - Automatically numbers existing and selected takes
--     (e.g. Base 01_suffix, Base 02_suffix)
--   - Ignores currently selected takes when scanning existing ones
--   - Appends META project markers after item end
--
--   ## Requirements
--   - SWS Extension (CF_GetClipboard)
--
--   ## Notes
--   - Designed for Soundminer / UCS workflows
--   - Safe to run multiple times
--
-- @changelog
--   v1.1.0
--   - Ignore selected takes when scanning for existing names
--   - Stable numbering logic
--   - META markers placed safely after item end


----------------------------------------------------------
-- Debug helper
----------------------------------------------------------
function Msg(param)
  reaper.ShowConsoleMsg(tostring(param) .. "\n")
end


----------------------------------------------------------
-- Marker offsets
----------------------------------------------------------
local META_AFTER_END_OFFSET = 0.0005
local META_SECOND_DELTA     = 0.001


----------------------------------------------------------
-- Escape Lua patterns
----------------------------------------------------------
local function escape_lua_pattern(s)
  return (s:gsub("([%%%^%$%(%)%.%[%]%*%+%-%?])", "%%%1"))
end


----------------------------------------------------------
-- Get clipboard text
----------------------------------------------------------
local clipboard = reaper.CF_GetClipboard("")
if not clipboard or clipboard == "" then
  reaper.MB("Clipboard is empty!", "Error", 0)
  return
end


----------------------------------------------------------
-- Extract Filename="..."
----------------------------------------------------------
local filename = clipboard:match('Filename="(.-)"')
if not filename then
  reaper.MB("Could not find Filename in clipboard!", "Error", 0)
  return
end


----------------------------------------------------------
-- Split base name and suffix
----------------------------------------------------------
local lastUnderscorePos = filename:match(".*()_")
local baseNoNum, suffix

if lastUnderscorePos then
  baseNoNum = filename:sub(1, lastUnderscorePos - 1)
  suffix    = filename:sub(lastUnderscorePos)
else
  baseNoNum = filename
  suffix    = ""
end

local escSuffix = escape_lua_pattern(suffix)


----------------------------------------------------------
-- Collect selected items and takes
----------------------------------------------------------
local selCount = reaper.CountSelectedMediaItems(0)
if selCount == 0 then
  reaper.MB("No items selected!", "Error", 0)
  return
end

local selectedItems = {}
local selectedTakes = {}

for i = 0, selCount - 1 do
  local item = reaper.GetSelectedMediaItem(0, i)
  local take = reaper.GetActiveTake(item)
  if take then
    selectedTakes[take] = true
    table.insert(selectedItems, { item = item, take = take })
  end
end


--------------------------------------------------------------------------------
-- Scan existing takes (IGNORE selected takes entirely)
--------------------------------------------------------------------------------
local highest        = 0
local plainTakes     = {}
local existingCount  = 0

local itemCount = reaper.CountMediaItems(0)
for i = 0, itemCount - 1 do
  local item = reaper.GetMediaItem(0, i)
  local take = reaper.GetActiveTake(item)

  if take and not selectedTakes[take] then
    local _, tname = reaper.GetSetMediaItemTakeInfo_String(
      take, "P_NAME", "", false
    )

    if tname == (baseNoNum .. suffix) then
      existingCount = existingCount + 1
      table.insert(plainTakes, { item = item, take = take })

    else
      local basePart, numStr =
        tname:match("^(.-) (%d+)" .. escSuffix .. "$")

      if basePart and basePart == baseNoNum then
        existingCount = existingCount + 1
        highest = math.max(highest, tonumber(numStr) or 0)
      end
    end
  end
end


reaper.Undo_BeginBlock()


-------------------------------------------------------------------
-- CASE 1: No matching takes anywhere else → keep plain name
-------------------------------------------------------------------
if existingCount == 0 and #selectedItems == 1 then
  local t = selectedItems[1]

  reaper.GetSetMediaItemTakeInfo_String(
    t.take, "P_NAME", filename, true
  )

  local itemStart = reaper.GetMediaItemInfo_Value(t.item, "D_POSITION")
  local itemEnd   = itemStart +
                    reaper.GetMediaItemInfo_Value(t.item, "D_LENGTH")

  local meta1Pos = itemEnd + META_AFTER_END_OFFSET

  reaper.AddProjectMarker(0, false, meta1Pos, 0, clipboard, -1)
  reaper.AddProjectMarker(
    0, false, meta1Pos + META_SECOND_DELTA, 0, "META", -1
  )

  reaper.Undo_EndBlock("Rename take (plain) + META", -1)
  return
end


-------------------------------------------------------------------
-- CASE 2: Convert existing plain takes (not selected)
-------------------------------------------------------------------
local nextIndex = highest

for _, t in ipairs(plainTakes) do
  nextIndex = nextIndex + 1
  local newname = string.format(
    "%s %02d%s", baseNoNum, nextIndex, suffix
  )

  reaper.GetSetMediaItemTakeInfo_String(
    t.take, "P_NAME", newname, true
  )
end


-------------------------------------------------------------------
-- CASE 3: Number selected takes and add META markers
-------------------------------------------------------------------
for _, t in ipairs(selectedItems) do
  nextIndex = nextIndex + 1

  local newname = string.format(
    "%s %02d%s", baseNoNum, nextIndex, suffix
  )

  reaper.GetSetMediaItemTakeInfo_String(
    t.take, "P_NAME", newname, true
  )

  local itemStart = reaper.GetMediaItemInfo_Value(t.item, "D_POSITION")
  local itemEnd   = itemStart +
                    reaper.GetMediaItemInfo_Value(t.item, "D_LENGTH")

  local meta1Pos = itemEnd + META_AFTER_END_OFFSET

  reaper.AddProjectMarker(0, false, meta1Pos, 0, clipboard, -1)
  reaper.AddProjectMarker(
    0, false, meta1Pos + META_SECOND_DELTA, 0, "META", -1
  )
end


reaper.Undo_EndBlock("Rename takes + META markers", -1)

