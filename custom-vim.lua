function makeprg_release()
  vim.o.makeprg = 'puledit build'
  vim.cmd("make")
end
