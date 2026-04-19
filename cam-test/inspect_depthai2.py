import depthai as dai
X = dai.node.internal.XLinkOut
print('XLinkOut mro:', X.__mro__)
print('hasNode', hasattr(dai.node, 'Node'))
print('issubclass', issubclass(X, dai.node.Node) if hasattr(dai.node, 'Node') else 'no Node')
