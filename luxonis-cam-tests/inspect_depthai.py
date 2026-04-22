import depthai as dai
print('dai version', dai.__version__)
print('Pipeline attrs', [n for n in dir(dai.Pipeline) if 'create' in n.lower() or 'xlink' in n.lower()])
print('dai.node attrs sample', [n for n in dir(dai.node) if 'xlink' in n.lower()][:50])
print('dai.node.internal attrs sample', [n for n in dir(dai.node.internal) if 'xlink' in n.lower()][:50])
print('has dai.node.XLinkOut', hasattr(dai.node, 'XLinkOut'))
print('has dai.node.internal.XLinkOut', hasattr(dai.node.internal, 'XLinkOut'))
print('XLinkOut class', getattr(dai.node.internal, 'XLinkOut', None))
print('XLinkOutHost class', getattr(dai.node.internal, 'XLinkOutHost', None))
