templates = dict(
    figure=dict(
        layout=dict(
            geo=dict(
                resolution=50,
                projection=dict(type='orthographic', scale=1.0),
                showcoastlines=False,
                showocean=True,
                landcolor="rgba(166,166,166,0.625)",
                oceancolor="rgb(222,222,222)",
                showlakes=False,
                showrivers=False,
                showcountries=False,
                lataxis=dict(showgrid=True),
                lonaxis=dict(showgrid=True)
            ),
            # title=dict(text='', x=0.5),
            autosize=True,
            margin=dict(l=0, r=0, t=75, b=75)
        )
    ),
    main_quant=dict(
        colorscale='Viridis',
        colorbar=dict(title='', ypad=0),
        marker=dict(line=dict(color='white', width=0.60), opacity=0.68),
        showscale=True,
        showlegend=False,
        hoverinfo='location+z+text'
    ),
    main_qual=dict(
        colorscale='Set3',
        colorbar=dict(title='COUNT', ypad=0),
        marker=dict(line=dict(color='white', width=0.60), opacity=0.68),
        showscale=False,
        showlegend=True,
        hoverinfo='location+z+text'
    ),
    region=dict(
        colorscale=[[0, 'rgba(255,255,255,0.525)'], [1, 'rgba(255,255,255,0.525)']],
        marker=dict(line=dict(color="rgba(0,0,0,1)", width=0.65)),
        legendgroup='regions',
        zmin=0,
        zmax=1,
        showlegend=False,
        showscale=False,
        hoverinfo='text'
    ),
    grid=dict(
        colorscale=[[0, 'rgba(255,255,255,0.01)'], [1, 'rgba(255,255,255,0.01)']],
        zmax=1,
        zmin=0,
        marker=dict(line=dict(color='rgba(255,255,255,1)', width=0.60), opacity=1),
        showlegend=False,
        showscale=False,
        hoverinfo='text',
        legendgroup='grids'
    ),
    point=dict(
        mode='markers+text',
        marker=dict(
            line=dict(color='black', width=0.3),
            color='white',
            symbol='circle-dot',
            size=6
        ),
        showlegend=False,
        textposition='top center',
        textfont=dict(color='Black', size=5)
    ),
    outline=dict(
        mode='lines',
        line=dict(color="black", width=1, dash='dash'),
        legendgroup='outlines',
        showlegend=False,
        hoverinfo='text'
    )
)


def get_template(name: str) -> dict:
    """Retrieves a template from the module.

    :param name: The name of the template
    :type name: str
    :return: The retrieved template
    :rtype: dict
    """
    try:
        return templates[name]
    except KeyError:
        raise ValueError('The specified template could not be found.')
